//! The turn-record store (issue #1252): the full text of every turn.
//!
//! Two tables, created by migration `055_turn_records.sql`. `turn_records` is
//! one row per turn - whose it was, which conversation, where it dispatched
//! and under which tool policy. `turn_round_records` is one row per round
//! within it, carrying the request exactly as sent, the reply, the tool calls
//! and their results.
//!
//! Both are personal data of the widest kind: a round row holds the assembled
//! system prompt, every injected block, the person's own words and whatever a
//! tool read on their behalf. Every statement here binds `current_user_id()`,
//! both tables enable their own row-level security in the migration, and both
//! are registered in `PERSONAL_DATA_TABLES`.
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so the `user_id` predicates written
//! here are the guard.
//!
//! ## Why the writes are shaped this way
//!
//! The turn row is an `ON CONFLICT DO NOTHING` insert, so a replay keeps the
//! first `started_at` - which is what retention measures from, and a later one
//! would quietly extend the window. The round row is an upsert on its
//! identity, and the tool results are a plain `UPDATE` over a row the round
//! write already made: an insert there would invent a round record with an
//! empty request, which reads as a turn that sent the model nothing.

pub mod retention;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolCall};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::llm::TokenUsage;
use desktop_assistant_core::ports::turn_record::{
    RoundRecord, RoundToolResults, StoredRound, StoredTurn, TurnRecord, TurnRecorder,
};
use sqlx::PgPool;
use sqlx::types::Json;

pub use retention::sweep_expired_turn_records;

/// Postgres-backed [`TurnRecorder`].
pub struct PgTurnRecordStore {
    pool: PgPool,
}

/// One `turn_records` row, as read.
#[derive(sqlx::FromRow)]
struct TurnRow {
    correlation_id: String,
    conversation_id: String,
    connection_id: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    tool_policy: String,
}

/// One `turn_round_records` row, as read.
#[derive(sqlx::FromRow)]
struct RoundRow {
    round: i32,
    request: Json<Vec<Message>>,
    response_text: String,
    response_tool_calls: Json<Vec<ToolCall>>,
    tool_results: Json<Vec<Message>>,
    token_usage: Option<Json<TokenUsage>>,
    error: Option<String>,
}

fn storage_error(e: sqlx::Error) -> CoreError {
    CoreError::Storage(e.to_string())
}

impl PgTurnRecordStore {
    /// Build a store over `pool`.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// One turn and every round of it, for the caller's own user.
    ///
    /// `None` when this user has no turn under `correlation_id` - which is
    /// also the answer another tenant gets for a turn that exists, because
    /// both statements bind the caller's own id and nothing here widens it.
    /// Reading across users is a capability of its own
    /// (`transport_dispatch::authz::READ_ANY_USER_TURN_RECORDS`) and no path
    /// in this crate holds it.
    pub async fn read_turn(&self, correlation_id: &str) -> Result<Option<StoredTurn>, CoreError> {
        let user_id = current_user_id();
        let turn: Option<TurnRow> = sqlx::query_as(
            "SELECT correlation_id, conversation_id, connection_id, provider, model, tool_policy \
             FROM turn_records \
             WHERE user_id = $1 AND correlation_id = $2",
        )
        .bind(user_id.as_str())
        .bind(correlation_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(storage_error)?;

        let Some(turn) = turn else {
            return Ok(None);
        };

        let rounds: Vec<RoundRow> = sqlx::query_as(
            "SELECT round, request, response_text, response_tool_calls, tool_results, \
                    token_usage, error \
             FROM turn_round_records \
             WHERE user_id = $1 AND correlation_id = $2 \
             ORDER BY round",
        )
        .bind(user_id.as_str())
        .bind(correlation_id)
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        Ok(Some(StoredTurn {
            turn: TurnRecord {
                correlation_id: turn.correlation_id,
                conversation_id: turn.conversation_id,
                connection_id: turn.connection_id,
                provider: turn.provider,
                model: turn.model,
                tool_policy: turn.tool_policy,
            },
            rounds: rounds
                .into_iter()
                .map(|row| StoredRound {
                    // The column is `INTEGER` because a round index is small
                    // and bounded by `MAX_TOOL_ROUNDS`; the domain calls it a
                    // `u32` because it can never be negative.
                    round: row.round.max(0) as u32,
                    request: row.request.0,
                    response_text: row.response_text,
                    response_tool_calls: row.response_tool_calls.0,
                    tool_results: row.tool_results.0,
                    usage: row.token_usage.map(|usage| usage.0),
                    error: row.error,
                })
                .collect(),
        }))
    }
}

#[async_trait::async_trait]
impl TurnRecorder for PgTurnRecordStore {
    async fn record_turn(&self, turn: TurnRecord) -> Result<(), CoreError> {
        let user_id = current_user_id();
        sqlx::query(
            "INSERT INTO turn_records \
                 (user_id, correlation_id, conversation_id, connection_id, provider, model, \
                  tool_policy) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             ON CONFLICT (user_id, correlation_id) DO NOTHING",
        )
        .bind(user_id.as_str())
        .bind(&turn.correlation_id)
        .bind(&turn.conversation_id)
        .bind(&turn.connection_id)
        .bind(&turn.provider)
        .bind(&turn.model)
        .bind(&turn.tool_policy)
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;
        Ok(())
    }

    async fn record_round(&self, round: RoundRecord) -> Result<(), CoreError> {
        let user_id = current_user_id();
        // Two things this statement will not do, and both are refusals rather
        // than defaults.
        //
        // `tool_results` is absent from the update list: this write happens
        // when the provider answers and the round's results arrive afterwards,
        // so naming it would erase them on a replay.
        //
        // The `WHERE` on the update is what stops one correlation id spanning
        // two conversations. The id is the CLIENT's to choose, so a client that
        // reuses one starts again at round 1 - and a plain upsert would then
        // overwrite the first conversation's round with the second's content
        // while the row went on naming the first. The foreign key cannot catch
        // it: an update that leaves the referencing columns alone re-checks
        // nothing. So the update declines instead, no row changes, and the
        // caller hears about it below.
        let written = sqlx::query(
            "INSERT INTO turn_round_records \
                 (user_id, correlation_id, conversation_id, round, request, response_text, \
                  response_tool_calls, token_usage, error) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
             ON CONFLICT (user_id, correlation_id, round) DO UPDATE SET \
                 request = EXCLUDED.request, \
                 response_text = EXCLUDED.response_text, \
                 response_tool_calls = EXCLUDED.response_tool_calls, \
                 token_usage = EXCLUDED.token_usage, \
                 error = EXCLUDED.error \
             WHERE turn_round_records.conversation_id = EXCLUDED.conversation_id",
        )
        .bind(user_id.as_str())
        .bind(&round.correlation_id)
        .bind(&round.conversation_id)
        .bind(i32::try_from(round.round).unwrap_or(i32::MAX))
        .bind(Json(&round.request))
        .bind(&round.response_text)
        .bind(Json(&round.response_tool_calls))
        .bind(round.usage.as_ref().map(Json))
        .bind(&round.error)
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;
        if written.rows_affected() == 0 {
            // The only way to affect no rows here: the conflicting row belongs
            // to another conversation. An insert affects one, and so does an
            // update whose guard holds. Reported rather than swallowed - the
            // record for this round does not exist, and a caller that heard
            // `Ok` would go on to attach results to a round that is not its
            // own.
            return Err(CoreError::Storage(format!(
                "turn {} round {} already belongs to another conversation; \
                 the round was not recorded",
                round.correlation_id, round.round
            )));
        }
        Ok(())
    }

    async fn record_round_results(&self, results: RoundToolResults) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let updated = sqlx::query(
            "UPDATE turn_round_records SET tool_results = $5 \
             WHERE user_id = $1 AND correlation_id = $2 AND conversation_id = $3 AND round = $4",
        )
        .bind(user_id.as_str())
        .bind(&results.correlation_id)
        .bind(&results.conversation_id)
        .bind(i32::try_from(results.round).unwrap_or(i32::MAX))
        .bind(Json(&results.results))
        .execute(&self.pool)
        .await
        .map_err(storage_error)?;
        if updated.rows_affected() == 0 {
            // Two ways to get here, and neither is worth a second WARN. Either
            // the round's own write failed - and the turn already carried that
            // failure to the log at warn - or the round exists under a
            // different conversation, which the foreign key already refused to
            // create. Both name the round so the pair can be read together.
            tracing::debug!(
                target: "turn_records",
                round = results.round,
                "no round record of this conversation to attach tool results to"
            );
        }
        Ok(())
    }
}
