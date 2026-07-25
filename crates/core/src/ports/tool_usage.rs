//! Per-conversation tool-usage cost aggregate (#599), the Context Inspector's
//! "what did this session actually spend?" view.
//!
//! Two axes matter independently, and a chart carrying only one is misleading:
//!
//! - **Frequency** — a tool called forty times is a signal in itself (a search
//!   loop, a retry storm), even if each result is tiny.
//! - **Payload** — a tool called twice that poured 40 KiB into context costs far
//!   more than one called forty times returning twenty bytes.
//!
//! So the aggregate carries both, plus the spread (`max_result_bytes`) that
//! separates "steadily chatty" from "one enormous dump". Callers slice by
//! whichever axis they are asking about rather than being handed a single
//! ranking baked in here.
//!
//! **Derived, not captured.** Everything here is an aggregation over rows that
//! already exist — assistant `tool_calls` joined to their `Role::Tool` results —
//! so it needs no new write path and no migration, works retroactively on every
//! conversation already stored, and cannot drift from reality the way a parallel
//! counter would.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;

/// What one tool cost a conversation.
///
/// "Cost" is deliberately reported in both bytes and estimated tokens: bytes are
/// the ground truth we can measure, tokens are what the context budget actually
/// spends, and the conversion is an estimate (see [`estimate_tokens`]) rather
/// than a provider tokenizer — so exposing both keeps the estimate honest and
/// auditable instead of presenting a derived number as fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolUsage {
    pub tool_name: String,
    /// Resolved tool namespace (`builtin`, or an MCP server) where known, so a
    /// caller can answer "which server is this session leaning on".
    pub namespace: Option<String>,
    /// Invocations the model *requested*. Counts failures and calls that were
    /// requested but never executed (cancelled turn, round exhaustion — #289):
    /// a tool called five times and failing five times reads as 5, which is the
    /// signal you want, not 0.
    pub call_count: u32,
    /// Result bytes still resident in the conversation.
    pub result_bytes: u64,
    /// Largest single resident result. Separates a steady trickle from one
    /// enormous dump — the case `result_bytes` alone hides.
    pub max_result_bytes: u64,
    /// Results whose content has been compacted away to the scratchpad (#240).
    ///
    /// Their ORIGINAL size is not recoverable: eviction replaces the content
    /// with a pointer and records nothing about what it displaced. So
    /// `result_bytes` is what a tool costs *now*, and this count is the honest
    /// marker that it once cost more. Recovering true peak cost needs the
    /// per-turn capture in #588; it is deliberately not guessed at here.
    pub evicted_results: u32,
    /// Message ordinal of the first / last call, so a caller can jump straight
    /// to where a tool entered the conversation and line usage up against the
    /// lifecycle timeline (#589).
    ///
    /// Ordinals rather than timestamps because `messages` carries no timestamp
    /// column — and for a conversation-scoped view an ordinal is the more useful
    /// handle anyway, since it addresses a position the UI can navigate to.
    pub first_ordinal: i32,
    pub last_ordinal: i32,
    /// Wall-clock of the first / last call, RFC3339, or `None` for a message
    /// predating UUIDv7 ids.
    ///
    /// Not a stored column: message ids ARE UUIDv7 (migration 005), whose first
    /// 48 bits are the creation time in milliseconds — so the timestamp is
    /// recovered from the id rather than requiring a migration, and works
    /// retroactively on every message already stored.
    pub first_used_at: Option<String>,
    pub last_used_at: Option<String>,
}

impl ToolUsage {
    /// Estimated tokens for the resident result bytes.
    ///
    /// Uses the same `chars / 4` rule as the context budget's default estimator
    /// (`ports::llm`), so the number a caller displays is comparable with the
    /// budget figures rather than being a second, differently-derived estimate.
    /// Bytes stand in for chars here — they agree for ASCII and over-estimate
    /// slightly for multi-byte text, which errs toward over-reporting cost.
    pub fn result_tokens(&self) -> u64 {
        estimate_tokens(self.result_bytes)
    }
}

/// The shared byte→token estimate. See [`ToolUsage::result_tokens`].
pub fn estimate_tokens(bytes: u64) -> u64 {
    bytes.div_ceil(4)
}

/// Outbound port for the tool-usage aggregate. Read-only by construction —
/// there is nothing to write, which is the point.
pub trait ToolUsageStore: Send + Sync {
    /// Aggregate tool usage for one conversation, ordered by `call_count`
    /// descending then `tool_name` for a stable tie-break.
    ///
    /// Scoped by the task-local `UserId` as well as `conversation_id`, so one
    /// user's histogram can never include another's rows. A conversation with no
    /// tool calls yields an empty vec, not an error.
    fn tool_usage(
        &self,
        conversation_id: &str,
    ) -> impl Future<Output = Result<Vec<ToolUsage>, CoreError>> + Send;
}

/// Boxed async closure for the aggregate across non-generic boundaries
/// (mirrors [`ConversationSearchFn`]).
///
/// [`ConversationSearchFn`]: super::conversation_search::ConversationSearchFn
pub type ToolUsageFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<Vec<ToolUsage>, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(call_count: u32, result_bytes: u64) -> ToolUsage {
        ToolUsage {
            tool_name: "t".into(),
            namespace: None,
            call_count,
            result_bytes,
            max_result_bytes: result_bytes,
            evicted_results: 0,
            first_ordinal: 0,
            last_ordinal: 0,
            first_used_at: None,
            last_used_at: None,
        }
    }

    #[test]
    fn token_estimate_matches_the_context_budget_rule() {
        // Must agree with `ports::llm`'s default estimator (chars/4, rounding
        // up) or the inspector's numbers won't be comparable with the budget's.
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(1), 1, "a partial token still costs one");
        assert_eq!(estimate_tokens(4), 1);
        assert_eq!(estimate_tokens(5), 2);
        assert_eq!(usage(1, 4000).result_tokens(), 1000);
    }

    #[test]
    fn frequency_and_payload_are_independent_axes() {
        // The whole reason both are carried: neither ranking implies the other,
        // so a caller must be able to slice either way.
        let chatty = usage(40, 40 * 20);
        let heavy = usage(2, 40 * 1024);
        assert!(chatty.call_count > heavy.call_count);
        assert!(
            heavy.result_bytes > chatty.result_bytes,
            "the infrequent tool is the expensive one; a count-only chart would \
             rank it last"
        );
    }
}
