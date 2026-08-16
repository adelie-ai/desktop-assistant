//! Postgres adapter for the per-turn context breakdown (#588).
//!
//! One row per turn, keyed on `(user_id, request_id)`. The key is the turn's
//! own correlation id, so the write is idempotent by construction: a retried or
//! re-driven turn replaces its row instead of adding a second record of the
//! same turn.
//!
//! ## The part names come from the enum, never from here
//!
//! The ten prompt parts have one definition, `PromptPart`, and both directions
//! of the JSONB column derive their keys from [`PromptPart::ALL`] and
//! [`PromptPart::as_label`]. Nothing in this file spells a part name. A part
//! this build cannot name is skipped on read rather than attributed somewhere,
//! because attributing it would report one part's cost under another's name
//! while every total still added up - the failure the record exists to prevent.
//!
//! The same holds for the budget tier: [`BudgetSource::as_label`] writes it and
//! [`BudgetSource::from_label`] reads it, and a label from no tier reads back as
//! "no tier" rather than as the fallback, because answering "universal
//! fallback" would report a curated budget as an unconfigured one.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::context_breakdown::{
    ContextBreakdown, ContextBreakdownStore, PromptBreakdown, PromptPart,
};
use desktop_assistant_core::ports::llm::BudgetSource;
use sqlx::PgPool;
use sqlx::Row;

// Both reads spell their SQL out as a plain string literal at the call site,
// column list and all, rather than sharing one built with `format!`. The static
// `user_id` audit (crates/storage/tests/audit_user_id_scoping.rs) extracts the
// string literal passed to a `sqlx::query...(` call and skips a call whose first
// argument is anything else - so a query assembled from a constant is a query
// the audit silently does not scan, and a later edit could drop the `user_id`
// predicate with the whole gate still green. The cost is one column list written
// twice; `context_breakdown_round_trips_every_recorded_field` and
// `context_breakdown_rows_are_retrievable_for_the_whole_conversation` read every
// field back through BOTH paths against a real database, so a list that drifted
// from the get fails there.

pub struct PgContextBreakdownStore {
    pool: PgPool,
}

impl PgContextBreakdownStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// The measured parts as a JSON object keyed by each part's stable label.
fn parts_to_json(parts: &PromptBreakdown) -> serde_json::Value {
    let measured = PromptPart::ALL
        .iter()
        .map(|part| (part.as_label().to_string(), part_value(parts, *part)));
    serde_json::Value::Object(measured.collect())
}

fn part_value(parts: &PromptBreakdown, part: PromptPart) -> serde_json::Value {
    serde_json::Value::Number(parts.tokens(part).into())
}

/// Rebuild the breakdown from a stored object.
///
/// A key this build cannot name is skipped, and a part the object does not
/// carry keeps the zero a part that did not render recorded in the first place.
/// A non-numeric or negative value is read as absent for the same reason:
/// storing a figure and reading a different one is worse than reading none.
fn parts_from_json(stored: &serde_json::Value, tool_count: i32) -> PromptBreakdown {
    let figures = PromptPart::ALL.into_iter().filter_map(|part| {
        stored
            .get(part.as_label())
            .and_then(serde_json::Value::as_u64)
            .map(|tokens| (part, tokens))
    });
    PromptBreakdown::from_parts(figures, tool_count.max(0) as usize)
}

/// Map one selected row to the domain record.
fn row_to_breakdown(row: &sqlx::postgres::PgRow) -> Result<ContextBreakdown, CoreError> {
    let read = |column: &str| -> CoreError {
        CoreError::Storage(format!("malformed context_breakdowns row: column {column}"))
    };
    let stored_parts: serde_json::Value = row
        .try_get("estimated_parts")
        .map_err(|_| read("estimated_parts"))?;
    let tool_count: i32 = row
        .try_get("advertised_tool_count")
        .map_err(|_| read("advertised_tool_count"))?;
    let provider_used_tokens: Option<i64> = row
        .try_get("provider_used_tokens")
        .map_err(|_| read("provider_used_tokens"))?;
    let budget_tokens: Option<i64> = row
        .try_get("budget_tokens")
        .map_err(|_| read("budget_tokens"))?;
    let budget_source: Option<String> = row
        .try_get("budget_source")
        .map_err(|_| read("budget_source"))?;
    let projected: i32 = row
        .try_get("projected_messages")
        .map_err(|_| read("projected_messages"))?;
    let recorded_at: chrono::DateTime<chrono::Utc> = row
        .try_get("recorded_at")
        .map_err(|_| read("recorded_at"))?;
    Ok(ContextBreakdown {
        request_id: row.try_get("request_id").map_err(|_| read("request_id"))?,
        conversation_id: row
            .try_get("conversation_id")
            .map_err(|_| read("conversation_id"))?,
        turn_ordinal: row
            .try_get("turn_ordinal")
            .map_err(|_| read("turn_ordinal"))?,
        model: row.try_get("model").map_err(|_| read("model"))?,
        // A negative count is not a count the writer can produce, so reading
        // one as absent is honest where clamping it to zero would not be: zero
        // is what a provider reporting nothing must never look like.
        provider_used_tokens: provider_used_tokens.and_then(|v| u64::try_from(v).ok()),
        budget_tokens: budget_tokens.and_then(|v| u64::try_from(v).ok()),
        budget_source: budget_source.as_deref().and_then(BudgetSource::from_label),
        compaction_active: row
            .try_get("compaction_active")
            .map_err(|_| read("compaction_active"))?,
        parts: parts_from_json(&stored_parts, tool_count),
        projected_messages: projected.max(0) as u32,
        recorded_at: Some(recorded_at.to_rfc3339()),
    })
}

impl ContextBreakdownStore for PgContextBreakdownStore {
    async fn record(&self, breakdown: &ContextBreakdown) -> Result<(), CoreError> {
        let user_id = current_user_id();
        // `ON CONFLICT ... DO UPDATE` on the turn's own key, so a re-drive of
        // one turn replaces its record instead of writing a second one.
        // `recorded_at` is deliberately left alone by the update: it says when
        // the turn's account was first filed, and a re-drive does not make the
        // turn newer.
        sqlx::query(
            "INSERT INTO context_breakdowns \
                 (user_id, request_id, conversation_id, turn_ordinal, model, \
                  provider_used_tokens, budget_tokens, budget_source, \
                  compaction_active, estimated_parts, advertised_tool_count, \
                  projected_messages) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) \
             ON CONFLICT (user_id, request_id) DO UPDATE SET \
                 conversation_id = EXCLUDED.conversation_id, \
                 turn_ordinal = EXCLUDED.turn_ordinal, \
                 model = EXCLUDED.model, \
                 provider_used_tokens = EXCLUDED.provider_used_tokens, \
                 budget_tokens = EXCLUDED.budget_tokens, \
                 budget_source = EXCLUDED.budget_source, \
                 compaction_active = EXCLUDED.compaction_active, \
                 estimated_parts = EXCLUDED.estimated_parts, \
                 advertised_tool_count = EXCLUDED.advertised_tool_count, \
                 projected_messages = EXCLUDED.projected_messages",
        )
        .bind(user_id.as_str())
        .bind(&breakdown.request_id)
        .bind(&breakdown.conversation_id)
        .bind(breakdown.turn_ordinal)
        .bind(&breakdown.model)
        .bind(breakdown.provider_used_tokens.map(|v| v as i64))
        .bind(breakdown.budget_tokens.map(|v| v as i64))
        .bind(breakdown.budget_source.map(|s| s.as_label()))
        .bind(breakdown.compaction_active)
        .bind(parts_to_json(&breakdown.parts))
        .bind(i32::try_from(breakdown.parts.tool_count()).unwrap_or(i32::MAX))
        .bind(i32::try_from(breakdown.projected_messages).unwrap_or(i32::MAX))
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn list(
        &self,
        conversation_id: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<ContextBreakdown>, CoreError> {
        let user_id = current_user_id();
        // Conversation order, with the correlation id as the tie-break so two
        // turns recorded at one ordinal still page deterministically rather
        // than swapping places between reads.
        let rows = sqlx::query(
            "SELECT request_id, conversation_id, turn_ordinal, model, \
                    provider_used_tokens, budget_tokens, budget_source, \
                    compaction_active, estimated_parts, advertised_tool_count, \
                    projected_messages, recorded_at \
             FROM context_breakdowns \
             WHERE user_id = $1 AND conversation_id = $2 \
             ORDER BY turn_ordinal ASC, request_id ASC \
             LIMIT $3 OFFSET $4",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(i64::from(limit))
        .bind(i64::from(offset))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        rows.iter().map(row_to_breakdown).collect()
    }

    async fn get(&self, request_id: &str) -> Result<Option<ContextBreakdown>, CoreError> {
        let user_id = current_user_id();
        let row = sqlx::query(
            "SELECT request_id, conversation_id, turn_ordinal, model, \
                    provider_used_tokens, budget_tokens, budget_source, \
                    compaction_active, estimated_parts, advertised_tool_count, \
                    projected_messages, recorded_at \
             FROM context_breakdowns \
             WHERE user_id = $1 AND request_id = $2",
        )
        .bind(user_id.as_str())
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        row.as_ref().map(row_to_breakdown).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_part_is_written_under_its_own_label_and_read_back_as_itself() {
        // The one mistake this column cannot survive: a figure landing under
        // another part's key. Every part carries a different number, so a
        // swapped pair is visible rather than hidden behind equal values.
        let written = PromptBreakdown::from_parts(
            PromptPart::ALL
                .iter()
                .enumerate()
                .map(|(i, part)| (*part, (i as u64 + 1) * 100)),
            5,
        );
        let json = parts_to_json(&written);
        for part in PromptPart::ALL {
            assert!(
                json.get(part.as_label()).is_some(),
                "`{}` is missing from the stored object; a part with no key is \
                 a part with no figure",
                part.as_label()
            );
        }
        let read = parts_from_json(&json, 5);
        assert_eq!(read, written);
    }

    #[test]
    fn a_part_that_did_not_render_is_stored_as_zero_rather_than_left_out() {
        // Zero here IS the measurement: the assembler always knows whether it
        // emitted a block, so an absent key could only mean the part went
        // unmeasured, which is the one thing a reader must be able to tell
        // apart from an empty block.
        let written = PromptBreakdown::from_parts([(PromptPart::System, 40)], 0);
        let json = parts_to_json(&written);
        assert_eq!(
            json.get(PromptPart::Pinned.as_label())
                .and_then(serde_json::Value::as_u64),
            Some(0)
        );
    }

    #[test]
    fn a_stored_key_this_build_cannot_name_is_skipped_rather_than_attributed() {
        // A row written by a later build carrying an eleventh part. Charging
        // its cost to a part this build does know would report the wrong part
        // as expensive, with every total still adding up.
        let mut json = parts_to_json(&PromptBreakdown::from_parts(
            [(PromptPart::Transcript, 400)],
            2,
        ));
        json.as_object_mut()
            .expect("the parts object")
            .insert("some_later_part".into(), serde_json::json!(9_999));

        let read = parts_from_json(&json, 2);
        assert_eq!(read.tokens(PromptPart::Transcript), 400);
        assert_eq!(
            read.total_tokens(),
            400,
            "the unknown figure is not folded in"
        );
    }

    #[test]
    fn a_malformed_part_figure_reads_as_unmeasured_rather_than_as_a_number() {
        let json = serde_json::json!({ "transcript": "lots", "system": -1, "pinned": 12 });
        let read = parts_from_json(&json, 0);
        assert_eq!(read.tokens(PromptPart::Transcript), 0);
        assert_eq!(read.tokens(PromptPart::System), 0);
        assert_eq!(read.tokens(PromptPart::Pinned), 12);
    }
}
