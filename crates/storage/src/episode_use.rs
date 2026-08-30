//! Postgres adapter for the episode use log (#1350).
//!
//! Two tables, created by `063_episode_use_log.sql`. `episode_use_stats` holds
//! one row per episode per user - the counters, the first-seen stamp and the
//! recent-use window - and `episode_offers` holds the offers standing in each
//! conversation. Both are bounded per episode by design, on the reasoning
//! [`desktop_assistant_core::domain::knowledge_use`] states for the knowledge
//! log.
//!
//! ## Every offer is checked against the store, not only scoped by it
//!
//! An id is not evidence that the caller owns the digest it names. The offer
//! write therefore selects the row out of `turn_digests` under the caller's own
//! scope and inserts from that select, rather than inserting the id it was
//! handed. An id this person does not own, or one whose digest is in the trash,
//! records nothing and is not an error.
//!
//! That is also what keeps the foreign key from turning a race into a failed
//! batch: an id whose digest went away between the block rendering and the
//! offer landing simply selects nothing.
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so these predicates are the guard.
//!
//! ## No situation and no marks
//!
//! Neither act has a writer, so neither has a table - see
//! [`desktop_assistant_core::ports::episode_use`]. A record therefore comes
//! back with no marks, and the reinforcement term reads offers and opens alone.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::knowledge_use::{KnowledgeUseRecord, RECENT_USE_WINDOW};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::episode_use::EpisodeUseLog;
use desktop_assistant_core::ports::knowledge_use::{MAX_STANDING_OFFERS, OfferScope, OfferSource};
use sqlx::PgPool;

use crate::knowledge_use::USE_LOG_READ_STATEMENT_TIMEOUT;

/// The episode use log, backed by Postgres.
pub struct PgEpisodeUseLog {
    pool: PgPool,
}

impl PgEpisodeUseLog {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// Drop ids no stored id can equal, so one bad id is a miss rather than a
/// failed batch.
///
/// Postgres `text` cannot hold a NUL byte, so an id carrying one names nothing.
/// Sent as a parameter it raises instead of missing, and takes every other id
/// in the batch with it.
fn storable(ids: Vec<String>) -> Vec<String> {
    ids.into_iter().filter(|id| !id.contains('\0')).collect()
}

/// One `episode_use_stats` row as it comes back.
#[derive(sqlx::FromRow)]
struct StatsRow {
    episode_id: String,
    offered_count: i64,
    opened_count: i64,
    first_seen_at: DateTime<Utc>,
    last_offered_at: Option<DateTime<Utc>>,
    recent_uses: Vec<DateTime<Utc>>,
}

impl EpisodeUseLog for PgEpisodeUseLog {
    async fn record_offered(
        &self,
        scope: OfferScope,
        episode_ids: Vec<String>,
    ) -> Result<usize, CoreError> {
        let episode_ids = storable(episode_ids);
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
            sqlx::query("DELETE FROM episode_offers WHERE user_id = $1 AND conversation_id = $2")
                .bind(user_id.as_str())
                .bind(&scope.conversation_id)
                .execute(&mut *tx)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        let written = if episode_ids.is_empty() {
            0
        } else {
            let offered = sqlx::query(
                "INSERT INTO episode_offers (user_id, conversation_id, episode_id, offered_at) \
                 SELECT $1, $3, td.id, NOW() \
                 FROM turn_digests td \
                 WHERE td.id = ANY($2) AND td.user_id = $1 AND td.deleted_at IS NULL \
                 ON CONFLICT (user_id, conversation_id, episode_id) DO UPDATE SET \
                     offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&episode_ids)
            .bind(&scope.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .rows_affected() as usize;

            // The counters and the first-seen stamp, which are per episode
            // rather than per conversation.
            sqlx::query(
                "INSERT INTO episode_use_stats \
                     (user_id, episode_id, offered_count, first_seen_at, last_offered_at) \
                 SELECT $1, td.id, 1, NOW(), NOW() \
                 FROM turn_digests td \
                 WHERE td.id = ANY($2) AND td.user_id = $1 AND td.deleted_at IS NULL \
                 ON CONFLICT (user_id, episode_id) DO UPDATE SET \
                     offered_count = episode_use_stats.offered_count + 1, \
                     last_offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&episode_ids)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            // The recall write above clears before it inserts, so on an
            // ordinary deployment this trim never fires. It bounds the case
            // nothing else does - a conversation whose deployment renders no
            // block, where nothing clears at a turn boundary - on the same rule
            // the knowledge log's writer follows.
            sqlx::query(
                "DELETE FROM episode_offers o \
                 WHERE o.user_id = $1 AND o.conversation_id = $2 \
                   AND o.ctid NOT IN ( \
                       SELECT k.ctid FROM episode_offers k \
                       WHERE k.user_id = $1 AND k.conversation_id = $2 \
                       ORDER BY k.offered_at DESC, k.episode_id \
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
        episode_ids: Vec<String>,
    ) -> Result<usize, CoreError> {
        let episode_ids = storable(episode_ids);
        if episode_ids.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Taking the offer down is what makes the count idempotent: a second
        // read of the same episode in the same turn finds no standing offer, so
        // a retried tool call adds nothing. The delete decides which ids count,
        // and the update below counts exactly those.
        let taken: Vec<String> = sqlx::query_scalar(
            "DELETE FROM episode_offers \
             WHERE user_id = $1 AND conversation_id = $2 AND episode_id = ANY($3) \
             RETURNING episode_id",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id)
        .bind(&episode_ids)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if !taken.is_empty() {
            sqlx::query(
                "UPDATE episode_use_stats SET \
                     opened_count = opened_count + 1, \
                     recent_uses = (ARRAY[NOW()] || recent_uses)[1:$3::int] \
                 WHERE user_id = $1 AND episode_id = ANY($2)",
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
        Ok(taken.len())
    }

    async fn records(
        &self,
        episode_ids: Vec<String>,
    ) -> Result<Vec<KnowledgeUseRecord>, CoreError> {
        let episode_ids = storable(episode_ids);
        if episode_ids.is_empty() {
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
            "SELECT episode_id, offered_count, opened_count, first_seen_at, \
                    last_offered_at, recent_uses \
             FROM episode_use_stats \
             WHERE user_id = $1 AND episode_id = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&episode_ids)
        .fetch_all(&mut *read)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        read.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(stats
            .into_iter()
            .map(|row| KnowledgeUseRecord {
                // The record's identifier field carries the digest's row id,
                // which is what names an episode everywhere the model can reach
                // one.
                entry_id: row.episode_id,
                offered_count: row.offered_count.max(0) as u64,
                opened_count: row.opened_count.max(0) as u64,
                // No tool marks an episode, so no table holds one. The
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
