//! Rebuild the episodic turn index from the stored transcript (#1349).
//!
//! Without this, the store holds only the turns captured since the change,
//! which is too few to measure anything against - and measurement before
//! behaviour is how the epic above this work is being run.
//!
//! ## What it rebuilds, and what it refuses to
//!
//! One digest per turn, built by the same
//! [`desktop_assistant_core::turn_capture::capture_turn`] the live path uses,
//! so a backfilled digest and a captured one are the same artifact. Subagent
//! conversations are left out on their reserved tag, exactly as the live path
//! leaves them out.
//!
//! A turn the harness already captured is never rewritten. The live stamp is
//! the turn's own record of what it took in; a derived one is a
//! reconstruction, and where the two could disagree the record wins. This is
//! also what makes the pass idempotent: the candidate query asks only for
//! turns with no digest, and the insert declines a conflict rather than
//! replacing what is there.
//!
//! ## Provenance is re-derived, never defaulted
//!
//! `after_outside_read` is recomputed from the persisted tool traffic rather
//! than assumed. The transcript stores every tool call with its name and every
//! tool result with its bytes, so replaying them through
//! [`TurnProvenance::observe_result`] answers exactly what the live turn's own
//! gate answered.
//!
//! One case cannot be answered that way: a tool result whose request is not in
//! the turn - a row whose `tool_call_id` names no call the range holds, which
//! is what a windowed or compacted transcript leaves behind. The tool's name
//! is gone, so nothing can say whether those bytes were an outside party's.
//! Such a turn is treated as though it HAD read outside content
//! ([`TurnProvenance::observe_unattributed_result`]), which is the direction
//! that fails safe, and the pass reports how many turns it had to treat that
//! way.
//!
//! ## How the turn ended is recognised, not assumed
//!
//! A turn that failed has no answer half worth keeping: its closing assistant
//! text is the provider's error or the notice that says the user stopped the
//! turn, and filing that as an answer makes a recallable record whose content
//! is an outage. The live path is told which exit it took; a backfill is not,
//! because the transcript records what was said and not how the turn ended.
//!
//! Neither blanket answer is acceptable. Calling every backfilled turn
//! answered re-files every historical outage as a reply. Calling every one
//! unanswered makes each digest carry a disclosure that is false for the turns
//! that were answered. So the ending is recognised instead, by the openings of
//! the notices the harness itself writes
//! ([`desktop_assistant_core::turn_capture::FAILED_TURN_NOTICE_PREFIXES`]),
//! which the producers and the recogniser share.

use std::collections::HashMap;

use desktop_assistant_core::domain::{Message, RESERVED_SUBAGENT_TAG, Role, ToolCall};
use desktop_assistant_core::tool_provenance::TurnProvenance;
use desktop_assistant_core::turn_capture::{capture_turn, derive_turn_ending};
use sqlx::PgPool;

/// How many conversations one pass claims at a time.
const CONVERSATION_BATCH: i64 = 32;

/// What one backfill pass did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TurnDigestBackfill {
    /// Conversations that held at least one undigested turn and were read.
    pub conversations_scanned: usize,
    /// Digests written. A turn already digested is not counted, because the
    /// pass did not write it.
    pub digests_written: usize,
    /// Digests whose `after_outside_read` could not be derived from the stored
    /// tool traffic and was therefore set to the safe value. Reported so the
    /// proportion is visible rather than silently folded into the total.
    pub provenance_underivable: usize,
}

impl TurnDigestBackfill {
    /// Whether the pass found nothing to do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.digests_written == 0
    }
}

/// One conversation with undigested turns.
#[derive(sqlx::FromRow)]
struct CandidateRow {
    id: String,
    user_id: String,
}

/// One stored message, as much of it as the digest needs.
#[derive(sqlx::FromRow)]
struct MessageRow {
    id: String,
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
    tool_call_id: Option<String>,
}

impl MessageRow {
    fn into_message(self) -> Message {
        let role = match self.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::System,
        };
        let mut message = Message::new(role, &self.content);
        message.id = self.id;
        if let Some(json) = self.tool_calls
            && let Ok(calls) = serde_json::from_value::<Vec<ToolCall>>(json)
        {
            message.tool_calls = calls;
        }
        message.tool_call_id = self.tool_call_id;
        message
    }
}

/// The provenance a finished turn had by its end, re-derived from what it
/// stored.
///
/// Answers the provenance and whether anything in the turn was
/// unattributable, so the caller can report the proportion rather than let it
/// vanish into the stamp.
#[must_use]
pub fn derive_turn_provenance(turn: &[Message]) -> (TurnProvenance, bool) {
    let names: HashMap<&str, &str> = turn
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .flat_map(|m| m.tool_calls.iter())
        .map(|call| (call.id.as_str(), call.name.as_str()))
        .collect();

    let mut provenance = TurnProvenance::new();
    let mut underivable = false;
    for message in turn.iter().filter(|m| m.role == Role::Tool) {
        match message.tool_call_id.as_deref().and_then(|id| names.get(id)) {
            Some(name) => {
                provenance.observe_result(name, &message.content);
            }
            None => {
                underivable = true;
                provenance.observe_unattributed_result();
            }
        }
    }
    (provenance, underivable)
}

/// Split a conversation's messages into turns, each starting at a user
/// message.
///
/// Anything before the first user message is not a turn and is dropped: it has
/// no opening message to key a digest on, and nothing in it is the person's
/// words.
#[must_use]
fn turns_of(messages: Vec<Message>) -> Vec<Vec<Message>> {
    let mut turns: Vec<Vec<Message>> = Vec::new();
    for message in messages {
        if message.role == Role::User {
            turns.push(vec![message]);
        } else if let Some(current) = turns.last_mut() {
            current.push(message);
        }
    }
    turns
}

/// Rebuild the turn index from the stored transcript.
///
/// `hard_withhold` is the operator setting the live capture reads: under it,
/// what a turn DERIVED after reading outside content is replaced by a
/// placeholder and what the person said is kept. Passing the deployment's own
/// setting is what stops a backfill writing text the live path would have
/// destroyed.
///
/// Runs across every user, so it takes each row's own `user_id` from the
/// transcript rather than from a request scope; there is none at boot.
///
/// Returns what the pass did, including how many turns it could not derive a
/// provenance stamp for.
pub async fn backfill_turn_digests(
    pool: &PgPool,
    hard_withhold: bool,
) -> Result<TurnDigestBackfill, String> {
    let mut outcome = TurnDigestBackfill::default();

    loop {
        // Conversations still holding a turn with no digest. A subagent's
        // conversation is excluded here as well as inside `capture_turn`,
        // because without it every pass would re-read the same subagent
        // conversations for ever and never write a row - the loop's own
        // termination depends on a claimed conversation dropping out of this
        // query.
        let candidates: Vec<CandidateRow> = sqlx::query_as(
            "SELECT DISTINCT c.id, c.user_id \
               FROM conversations c \
               JOIN messages m \
                 ON m.conversation_id = c.id AND m.user_id = c.user_id \
               LEFT JOIN turn_digests d \
                 ON d.conversation_id = m.conversation_id \
                AND d.opening_message_id = m.id \
              WHERE m.role = 'user' \
                AND d.id IS NULL \
                AND NOT (c.tags @> ARRAY[$1]::text[]) \
              ORDER BY c.id \
              LIMIT $2",
        )
        .bind(RESERVED_SUBAGENT_TAG)
        .bind(CONVERSATION_BATCH)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

        if candidates.is_empty() {
            break;
        }

        let written_before = outcome.digests_written;
        for candidate in &candidates {
            outcome.conversations_scanned += 1;
            let rows: Vec<MessageRow> = sqlx::query_as(
                "SELECT id, role, content, tool_calls, tool_call_id \
                   FROM messages \
                  WHERE user_id = $1 AND conversation_id = $2 \
                  ORDER BY ordinal",
            )
            .bind(&candidate.user_id)
            .bind(&candidate.id)
            .fetch_all(pool)
            .await
            .map_err(|e| e.to_string())?;

            let messages: Vec<Message> = rows.into_iter().map(MessageRow::into_message).collect();
            for turn in turns_of(messages) {
                let (provenance, underivable) = derive_turn_provenance(&turn);
                // The conversation is known not to be a subagent's, so no tag
                // is passed; the exclusion above is what already decided it.
                let Some(digest) = capture_turn(
                    &turn,
                    0,
                    provenance,
                    hard_withhold,
                    derive_turn_ending(&turn),
                    &[],
                ) else {
                    continue;
                };
                let inserted = sqlx::query(
                    "INSERT INTO turn_digests \
                         (id, user_id, conversation_id, opening_message_id, content, \
                          after_outside_read) \
                     VALUES ($1, $2, $3, $4, $5, $6) \
                     ON CONFLICT (user_id, conversation_id, opening_message_id) \
                     DO NOTHING",
                )
                .bind(uuid::Uuid::now_v7().to_string())
                .bind(&candidate.user_id)
                .bind(&candidate.id)
                .bind(&digest.opening_message_id)
                .bind(&digest.content)
                .bind(digest.after_outside_read)
                .execute(pool)
                .await
                .map_err(|e| e.to_string())?;

                if inserted.rows_affected() > 0 {
                    outcome.digests_written += 1;
                    if underivable {
                        outcome.provenance_underivable += 1;
                    }
                }
            }
        }

        // A batch that claimed conversations and wrote nothing would be
        // claimed again for ever, because the candidate query would answer
        // with the same rows. Stopping and saying so is the only safe reading
        // of that: the alternative is a boot task that never finishes.
        if outcome.digests_written == written_before {
            tracing::warn!(
                conversations = candidates.len(),
                "turn-digest backfill claimed conversations but wrote no digest; stopping \
                 rather than re-reading the same rows"
            );
            break;
        }
    }

    if outcome.provenance_underivable > 0 {
        tracing::info!(
            written = outcome.digests_written,
            underivable = outcome.provenance_underivable,
            "turn-digest backfill could not derive a provenance stamp for some turns; \
             they are recorded as having read outside content, which is the safe direction"
        );
    }

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::{derive_turn_provenance, turns_of};
    use desktop_assistant_core::domain::{Message, Role, ToolCall};

    fn user(text: &str) -> Message {
        Message::new(Role::User, text)
    }

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, text)
    }

    fn requested(id: &str, name: &str) -> Message {
        Message::assistant_with_tool_calls(vec![ToolCall::new(id, name, "{}")])
    }

    #[test]
    fn a_conversation_splits_into_one_turn_per_user_message() {
        let messages = vec![
            assistant("a stray reply before anybody asked anything"),
            user("first"),
            assistant("one"),
            user("second"),
            requested("c1", "read_file"),
            Message::tool_result("c1", "bytes"),
            assistant("two"),
        ];
        let turns = turns_of(messages);
        assert_eq!(turns.len(), 2, "two prompts, two turns");
        assert_eq!(turns[0][0].content, "first");
        assert_eq!(turns[1][0].content, "second");
        assert_eq!(turns[1].len(), 4);
    }

    /// A turn whose tool traffic is all attributable derives the same answer
    /// the live gate gave, in both directions.
    #[test]
    fn provenance_is_derived_from_the_tool_that_produced_each_result() {
        let clean = vec![
            user("check the deploy"),
            requested("c1", "read_file"),
            Message::tool_result("c1", "the deploy notes"),
            assistant("it is fine"),
        ];
        let (provenance, underivable) = derive_turn_provenance(&clean);
        assert!(!provenance.ingested_external(), "read_file is trusted");
        assert!(!underivable);

        let tainted = vec![
            user("what does that page say"),
            requested("c1", "web_fetch"),
            Message::tool_result("c1", "<html>the page</html>"),
            assistant("it says something"),
        ];
        let (provenance, underivable) = derive_turn_provenance(&tainted);
        assert!(
            provenance.ingested_external(),
            "web_fetch returns outside content"
        );
        assert!(
            !underivable,
            "every result in this turn names a call the turn holds, so nothing \
             here was guessed at"
        );
    }

    /// A result whose request is not in the turn cannot be attributed, so it
    /// is treated as outside content rather than defaulted to trusted.
    #[test]
    fn an_unattributable_result_fails_safe() {
        let turn = vec![
            user("carry on"),
            Message::tool_result("gone", "whatever this was"),
            assistant("done"),
        ];
        let (provenance, underivable) = derive_turn_provenance(&turn);
        assert!(underivable, "the tool that produced it is not in the turn");
        assert!(
            provenance.ingested_external(),
            "an unattributable result must fail safe"
        );
    }
}
