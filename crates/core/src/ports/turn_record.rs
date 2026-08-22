//! What a turn actually said: the request as sent, the reply, the tool calls
//! and their results.
//!
//! The prompt breakdown measures what a turn's context weighed. This
//! records what it contained. The two answer different questions and only the
//! second answers "why did the assistant do that": a turn's system prompt, its
//! `[Recall]` block, its scratchpad injection and its post-eviction window
//! exist in memory for one provider call and are then gone, so the exact bytes
//! the model was shown are otherwise unrecoverable.
//!
//! ## A store, not a subscriber
//!
//! This is a record. It is not a tracing layer, it is not a log, and nothing
//! here writes to the console. A record is queryable, access-controlled and
//! retained on a schedule; a log line is none of those. The daemon's spans
//! keep their own job - latency and correlation - and carry no content.
//!
//! ## Keyed by the turn's own correlation id
//!
//! [`turn_correlation_id`] answers the id every record is filed under: the
//! value the client's own event stream carries, so a person quoting a reply
//! and the store reading it use one identifier.
//!
//! That value is usually the trace id too, because a turn nobody handed a
//! trace derives one from it. It is NOT the trace id when a caller forwarded a
//! `traceparent` to be continued - the trace is then the caller's and the two
//! differ, which is what
//! [`crate::ports::turn_telemetry::resolve_turn_trace`] is for. The record
//! follows the client's id in both cases, because that is the one a person can
//! actually quote.
//!
//! ## Three writes, and why they are separate
//!
//! A turn has many exits - an answer, a cancellation, a provider error, an
//! exhausted round budget - so nothing is held in memory until the end. Each
//! write lands as soon as its subject is known:
//!
//! 1. [`TurnRecorder::record_turn`], once, before the first round. A turn that
//!    then fails still has a record saying it happened and where it dispatched.
//! 2. [`TurnRecorder::record_round`], as soon as the provider answers or
//!    fails. This carries the request exactly as sent.
//! 3. [`TurnRecorder::record_round_results`], when the round's tool calls have
//!    resolved - and again on the cancellation exits inside the round, with
//!    what resolved before it stopped. Those calls committed their side
//!    effects, so recording nothing would say no tool ran when one did. A
//!    round that answered without calling anything never makes this call at
//!    all, and neither does one that was stopped before its first call: an
//!    empty write carries no information and would erase what a previous
//!    attempt under the same id recorded.
//!
//! A turn that spends its whole tool budget makes one further provider call -
//! the wind-down that turns an exhausted turn into a closing the person can
//! read - and it is recorded as a round one past the loop's last. Its request
//! exists nowhere else: the wrap-up instruction it carries is dropped before
//! the reply is persisted.
//!
//! Every one of them is idempotent: the same turn and the same round written
//! twice leave the store in the state one write would.
//!
//! ## What this is allowed to cost
//!
//! Nothing, when no recorder is wired. The turn loop clones the assembled
//! request only when [`ConversationHandler::with_turn_recorder`] has installed
//! one, so a daemon with capture off pays neither the clone nor the write.
//!
//! [`ConversationHandler::with_turn_recorder`]: crate::service::ConversationHandler::with_turn_recorder

use std::sync::Arc;

use crate::CoreError;
use crate::domain::{Message, ToolCall};
use crate::ports::llm::TokenUsage;
use crate::ports::turn_telemetry::{current_request_id, current_turn_trace};

/// The id every record of this turn is filed under.
///
/// The turn's correlation id where the transport supplied one, which is what
/// the client already correlates its own event stream by. A turn that reached
/// the loop by another door - an agent run, a scheduled job, a test - has no
/// such id, so the answer is that turn's trace id spelled as a uuid: the same
/// 16 bytes, one spelling, and still the value a trace backend indexes.
///
/// The two coincide on a turn nobody handed a trace, and diverge on one whose
/// caller forwarded a `traceparent` to be continued. The client's own id wins
/// there, because it is the one a person can quote.
///
/// A caller outside any turn scope gets a fresh uuid rather than a shared
/// sentinel, so two unrelated background writes can never collide on one key.
pub fn turn_correlation_id() -> String {
    if let Some(request_id) = current_request_id() {
        return request_id;
    }
    current_turn_trace()
        .and_then(|trace| uuid::Uuid::parse_str(&trace.trace.trace_id().to_hex()).ok())
        .unwrap_or_else(uuid::Uuid::new_v4)
        .to_string()
}

/// One turn, as it was dispatched.
///
/// The metadata a reader needs before it opens a single round: whose turn it
/// was, which conversation it belongs to, where it went, and what the tool
/// policy resolved to for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRecord {
    /// The turn's correlation id - see [`turn_correlation_id`].
    pub correlation_id: String,
    /// The conversation this turn is part of.
    ///
    /// Carried on the turn record *and* on every round record. The profiler
    /// this replaces wrote entries with neither a user nor a conversation on
    /// them, which made a file full of prompts unattributable and therefore
    /// unusable.
    pub conversation_id: String,
    /// The configured connection the turn dispatched through, where the daemon
    /// resolved one.
    pub connection_id: Option<String>,
    /// The connector kind behind that connection - `anthropic`, `ollama` and
    /// so on.
    pub provider: Option<String>,
    /// The model id the turn pinned for this dispatch.
    pub model: Option<String>,
    /// The tool policy this turn resolved to, as its stable spelling.
    pub tool_policy: String,
}

/// One round of a turn, as it was sent and as it came back.
///
/// `round` is one-based, matching the round span and the round's log line, so
/// a record and a trace read the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundRecord {
    /// The turn this round belongs to - see [`turn_correlation_id`].
    pub correlation_id: String,
    /// The conversation this round belongs to. On the round as well as on the
    /// turn, so no single record is unattributable on its own.
    pub conversation_id: String,
    /// Which round of the turn this is, one-based.
    pub round: u32,
    /// The request exactly as handed to the connector: every message, with its
    /// role, in the order it was sent, including the system prompt and every
    /// injected block.
    ///
    /// This is the assembled prompt and not the conversation. The two differ
    /// on every turn: the window is trimmed, tool results are projected down
    /// to their heads, and the injected blocks exist nowhere else.
    pub request: Vec<Message>,
    /// The reply text the provider streamed, whole.
    pub response_text: String,
    /// The tool calls the model asked for, with their arguments as the model
    /// wrote them.
    pub response_tool_calls: Vec<ToolCall>,
    /// What the provider reported this round cost, where it reported anything.
    pub usage: Option<TokenUsage>,
    /// Why the round failed, where it did. `None` is a round the provider
    /// answered.
    pub error: Option<String>,
}

/// What one round's tool calls returned, as the rows the turn stored.
///
/// The results are [`Message`]s with [`crate::domain::Role::Tool`], which is
/// what the turn appends to the conversation and what the next round reads.
/// Storing them as anything else would record a second version of the truth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundToolResults {
    /// The turn these results belong to.
    pub correlation_id: String,
    /// The conversation they belong to, for the same reason the round record
    /// carries one.
    pub conversation_id: String,
    /// The round whose calls produced them, one-based.
    pub round: u32,
    /// One row per resolved call, in the order the round resolved them.
    pub results: Vec<Message>,
}

/// One round as it reads back out of the store.
///
/// The write side splits a round across [`RoundRecord`] and
/// [`RoundToolResults`] because the two are known at different moments. A
/// reader has no such problem, so it gets the round whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredRound {
    /// Which round of the turn this is, one-based.
    pub round: u32,
    /// The request exactly as it was sent - see [`RoundRecord::request`].
    pub request: Vec<Message>,
    /// The reply text, whole.
    pub response_text: String,
    /// The tool calls the model asked for.
    pub response_tool_calls: Vec<ToolCall>,
    /// What those calls returned, as the rows the turn stored. Empty for a
    /// round that answered without calling anything, and for one that was
    /// stopped before its first call resolved.
    pub tool_results: Vec<Message>,
    /// What the provider reported this round cost.
    pub usage: Option<TokenUsage>,
    /// Why the round failed, where it did.
    pub error: Option<String>,
}

/// One turn as it reads back out of the store: its dispatch, then its rounds
/// in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredTurn {
    /// Whose turn it was, which conversation, and where it dispatched.
    pub turn: TurnRecord,
    /// Every round of the turn, ascending.
    pub rounds: Vec<StoredRound>,
}

/// Where a turn's full text is written.
///
/// Implemented by the storage adapter and installed on the turn loop by the
/// daemon when turn capture is on. Nothing is installed when it is off, and
/// the loop then does no extra work at all.
///
/// Every method is idempotent (see the module header): a retry, a redelivery
/// or a second daemon replaying the same turn leaves one record.
///
/// A failing write must not fail the turn. The caller logs it and carries on -
/// a debugging record is worth less than the answer a person asked for.
#[async_trait::async_trait]
pub trait TurnRecorder: Send + Sync {
    /// Record that a turn started, and where it dispatched.
    async fn record_turn(&self, turn: TurnRecord) -> Result<(), CoreError>;

    /// Record one round's request and what came back.
    async fn record_round(&self, round: RoundRecord) -> Result<(), CoreError>;

    /// Record what one round's tool calls returned.
    async fn record_round_results(&self, results: RoundToolResults) -> Result<(), CoreError>;
}

/// A recorder the turn loop holds, or none at all.
///
/// `None` is the shape a daemon with capture off runs, and it is also what
/// every test and background job gets unless it asks for otherwise.
pub type SharedTurnRecorder = Arc<dyn TurnRecorder>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::turn_telemetry::{TurnTrace, with_request_id, with_turn_trace};

    const REQUEST_ID: &str = "11111111-2222-4333-8444-555555555555";

    #[tokio::test]
    async fn the_correlation_id_is_the_turns_own_request_id() {
        let seen = with_request_id(REQUEST_ID.to_string(), async { turn_correlation_id() }).await;
        assert_eq!(seen, REQUEST_ID);
    }

    #[tokio::test]
    async fn a_turn_with_no_request_id_is_keyed_by_its_trace() {
        // An agent run, a scheduled job and a test reach the loop with no
        // correlation id. Keying such a turn by something unrelated to its
        // trace would leave its records unjoinable to anything else about it.
        let trace = TurnTrace::minted(None, "conv-1");
        let expected = trace.trace.trace_id().to_hex();
        let seen = with_turn_trace(Some(trace), async { turn_correlation_id() }).await;
        assert_eq!(seen.replace('-', ""), expected);
    }

    #[tokio::test]
    async fn two_unscoped_writes_never_share_a_key() {
        // A shared sentinel would file two unrelated turns under one id, and
        // the second would read as more rounds of the first.
        assert_ne!(turn_correlation_id(), turn_correlation_id());
    }
}
