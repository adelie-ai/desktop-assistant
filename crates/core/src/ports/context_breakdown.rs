//! One durable record per turn of what filled its prompt (#588).
//!
//! ## The question this answers, and why a record was needed for it
//!
//! The per-part measurement already happens on every turn: the assembler fills
//! a [`PromptBreakdown`] as it lays each block out, the turn span carries the
//! figures, and the metrics facade accumulates them. All of it is then dropped.
//! So an operator can see that a turn cost 40k, and can see the split for a
//! turn they are watching live, but cannot ask what filled the prompt of the
//! turn that went wrong an hour ago - or of any earlier turn in the same
//! conversation.
//!
//! The budget tier is dropped the same way. The daemon resolves every budget as
//! a user override, a connector's curated table, the universal fallback or a
//! learned cap, and nothing leaves the daemon. A real curated 200k and a silent
//! universal-fallback 200k are the same number and a different situation.
//!
//! This port joins the two and keeps them, one row per turn, scoped to the user
//! and the conversation, so the whole conversation is inspectable.
//!
//! ## Two measurements, side by side, never summed
//!
//! [`ContextBreakdown::parts`] is **estimated**, counted with the estimator the
//! context budget itself uses, and [`ContextBreakdown::provider_used_tokens`]
//! is what the provider reported for the same prompt. They are two measurements
//! of one thing, taken by different counters, and they do not agree. Nothing
//! here adds them, derives one from the other, or fills one in from the other:
//! a reader compares them, and the difference between them is itself the
//! signal.
//!
//! The absence rules differ for the same reason, and both are deliberate. A
//! part that did not render records zero, because the assembler always knows
//! whether it emitted a block. A provider that declined to report records
//! nothing at all, because a zero there would invent a measurement.
//!
//! ## One definition of the parts
//!
//! [`PromptPart`] and [`PromptBreakdown`] are the assembler's own types,
//! re-exported here rather than copied. Two lists of parts that can drift is
//! the defect this record exists to prevent: a second list would keep
//! reporting, and would report the wrong part's cost under the right name.
//! Everything downstream - the wire view, the stored row - derives its part
//! names from [`PromptPart::ALL`] and [`PromptPart::as_label`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::ports::llm::BudgetSource;

pub use crate::telemetry::prompt::{PromptBreakdown, PromptPart};

/// What filled one turn's prompt, and what the turn was allowed to spend.
///
/// Keyed by `request_id`: the correlation id the client already sees, which is
/// also the turn's trace id, so one value moves from a client's event stream to
/// this record to a trace backend with no mapping table in between.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextBreakdown {
    /// The turn's correlation id. Unique per turn, and the key a caller reads
    /// one record back by.
    pub request_id: String,
    /// The conversation the turn ran in.
    pub conversation_id: String,
    /// Where the turn begins in the conversation: the message ordinal its user
    /// prompt took. Lets a reader line the record up against the transcript,
    /// the way the tool-usage view's ordinals do.
    pub turn_ordinal: i32,
    /// The model the turn actually ran on, as the route resolved it.
    pub model: String,
    /// Prompt tokens the provider reported for the prompt [`Self::parts`]
    /// describes.
    ///
    /// `None` when the provider reported no count, which is not the same as
    /// zero: see the module header.
    pub provider_used_tokens: Option<u64>,
    /// The input-token budget this turn resolved, or `None` when no budget was
    /// installed (a background job, a test).
    pub budget_tokens: Option<u64>,
    /// Which tier produced [`Self::budget_tokens`]. Absent with it.
    pub budget_source: Option<BudgetSource>,
    /// Whether proactive compaction ran on this turn: the window was shrunk
    /// under token pressure and the dropped range summarised.
    pub compaction_active: bool,
    /// What each part of the prompt cost, in estimated tokens, plus how many
    /// tool schemas the prompt advertised. The assembler's own figures.
    pub parts: PromptBreakdown,
    /// How many messages the turn read as something other than their stored
    /// content: a compaction pointer for a result an earlier step distilled
    /// into a note, the head of a result too large to read inline, or a
    /// truncation notice written by overflow recovery.
    ///
    /// A count of what the transcript part is NOT charging for. The stored
    /// transcript still holds every byte, so this is the gap between what the
    /// conversation holds and what the model read.
    pub projected_messages: u32,
    /// When the row was written, RFC3339. `None` on a record that has not been
    /// stored yet - the store assigns it.
    pub recorded_at: Option<String>,
}

impl ContextBreakdown {
    /// Every measured part summed: what the estimator says this prompt cost.
    ///
    /// Not comparable with [`Self::provider_used_tokens`] as an equality, and
    /// not a component of it. See the module header.
    pub fn estimated_total_tokens(&self) -> u64 {
        self.parts.total_tokens()
    }

    /// How many tool schemas the prompt advertised. A count, not a token
    /// figure.
    pub fn advertised_tool_count(&self) -> u32 {
        u32::try_from(self.parts.tool_count()).unwrap_or(u32::MAX)
    }
}

/// Outbound port for the per-turn record: one write at the end of a turn, two
/// reads.
pub trait ContextBreakdownStore: Send + Sync {
    /// Write one turn's record.
    ///
    /// Idempotent on `(user_id, request_id)`: a repeat of the same turn's write
    /// replaces the row rather than adding a second one, so a retried or
    /// re-driven turn cannot double-count itself.
    fn record(
        &self,
        breakdown: &ContextBreakdown,
    ) -> impl Future<Output = Result<(), CoreError>> + Send;

    /// One conversation's records, oldest turn first, `limit` from `offset`.
    ///
    /// Conversation order rather than newest-first because the pages have to
    /// stay stable while the conversation grows: a new turn appends to the end,
    /// so it changes no page a caller has already read. Scoped by the
    /// task-local user id as well as `conversation_id`.
    fn list(
        &self,
        conversation_id: &str,
        limit: u32,
        offset: u32,
    ) -> impl Future<Output = Result<Vec<ContextBreakdown>, CoreError>> + Send;

    /// One turn's record by its correlation id, or `None`.
    ///
    /// Scoped by the task-local user id: a correlation id is not a capability,
    /// so another user's turn reads as absent.
    fn get(
        &self,
        request_id: &str,
    ) -> impl Future<Output = Result<Option<ContextBreakdown>, CoreError>> + Send;
}

/// Boxed async closure for the write, across non-generic boundaries.
///
/// The turn loop holds one of these rather than a store, so `core` records
/// without depending on any storage crate, and a deployment with no database
/// records nothing instead of failing turns.
pub type ContextBreakdownRecordFn = Arc<
    dyn Fn(ContextBreakdown) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for the conversation read: `(conversation_id, limit,
/// offset)`.
pub type ContextBreakdownListFn = Arc<
    dyn Fn(
            String,
            u32,
            u32,
        )
            -> Pin<Box<dyn Future<Output = Result<Vec<ContextBreakdown>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for the single-turn read, keyed by correlation id.
pub type ContextBreakdownGetFn = Arc<
    dyn Fn(
            String,
        )
            -> Pin<Box<dyn Future<Output = Result<Option<ContextBreakdown>, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    fn breakdown() -> ContextBreakdown {
        ContextBreakdown {
            request_id: "r1".into(),
            conversation_id: "c1".into(),
            turn_ordinal: 0,
            model: "m".into(),
            provider_used_tokens: Some(41_000),
            budget_tokens: Some(200_000),
            budget_source: Some(BudgetSource::ConnectorTable),
            compaction_active: false,
            parts: PromptBreakdown::from_parts([(PromptPart::Transcript, 400)], 3),
            projected_messages: 0,
            recorded_at: None,
        }
    }

    #[test]
    fn the_estimated_total_is_the_parts_and_never_the_providers_count() {
        // The one arithmetic mistake this record cannot survive: reporting the
        // provider's figure as though the parts summed to it.
        let record = breakdown();
        assert_eq!(record.estimated_total_tokens(), 400);
        assert_ne!(
            Some(record.estimated_total_tokens()),
            record.provider_used_tokens
        );
    }

    #[test]
    fn the_advertised_tool_count_is_a_count_and_not_a_token_figure() {
        let record = breakdown();
        assert_eq!(record.advertised_tool_count(), 3);
        assert_eq!(
            record.parts.tokens(PromptPart::ToolSchemas),
            0,
            "no schema cost was measured, and the count does not stand in for one"
        );
    }
}
