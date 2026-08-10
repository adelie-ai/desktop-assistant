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

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::knowledge_use::{KnowledgeUseRecord, RECENT_USE_WINDOW};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge_use::{MAX_STANDING_OFFERS, OfferScope, OfferSource};
use desktop_assistant_core::ports::skill_use::SkillUseLog;
use sqlx::PgPool;

use crate::knowledge_use::USE_LOG_READ_STATEMENT_TIMEOUT;

/// The skill use log, backed by Postgres.
pub struct PgSkillUseLog {
    pool: PgPool,
}

impl PgSkillUseLog {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
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
        Ok(taken.len())
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
