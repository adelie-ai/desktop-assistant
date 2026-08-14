//! The turn's ledger of the tool calls it has already dispatched (#1301).
//!
//! ## The problem
//!
//! Nothing in the turn loop used to notice that the model had already made a
//! given tool call. An identical `(tool, arguments)` pair ran again, returned
//! the same bytes again, and those bytes were appended to the context again.
//! On its own that is waste. Together with context eviction it is a loop with
//! an engine: a large result forces an eviction, the evicted result is the one
//! the model still needs, so the model fetches it again and forces the next
//! eviction. The loop is stable. It does not converge, and it does not fail -
//! it ends when the round cap fires or the model happens to answer.
//!
//! ## The rule
//!
//! Per key, the ledger records how many times the call has executed in this
//! turn, the message id of the first result, and whether every result so far
//! has been byte-identical.
//!
//! - First call - execute.
//! - Second call - execute, and tell the model in the result it gets back that
//!   this exact call already ran, naming the message the first result is
//!   stored under. A model that cannot tell "I did this" from "I did not" will
//!   do it again.
//! - Third and later call - when every prior execution returned byte-identical
//!   output, do not execute. Answer with a pointer to the first result, which
//!   [`crate::ports::transcript::TRANSCRIPT_GET_TOOL`] reads back (#1226).
//!   When the prior outputs differed, execute: the tool varies with time.
//!
//! ## Why the tool's own output is the evidence
//!
//! Taken literally, "a repeat does not execute" breaks legitimate repetition.
//! This rule distinguishes the two cases from evidence the tool supplies
//! itself, so it needs no tool taxonomy, no annotations and no allowlist:
//!
//! - A poll whose value has already changed once is never suppressed.
//! - A file re-read after a write is never suppressed, when the write landed
//!   before the second read: the write changed the bytes, so the two runs
//!   differ.
//!
//! ## What the rule does not hold, stated plainly
//!
//! The evidence is gathered from the first two runs and never gathered again.
//! A suppressed call does not execute, so it records nothing, so `all_identical`
//! can never go back to false. Suppression is therefore terminal for the rest
//! of the turn, and three cases fall outside the rule:
//!
//! - **A value that changes late.** Two runs that answer the same, then a third
//!   that would answer differently. A subagent poll reads `running` twice
//!   before the child completes, and a file read twice before the write is the
//!   same shape. The model is told to read the first result back, and the first
//!   result is now stale.
//! - **A failure that is fixed mid-turn.** An error is recorded like any other
//!   output, so a server that is restarting answers identically twice and the
//!   third call is answered from the transcript - even after the model has
//!   repaired the cause.
//! - **A side effect that leaves no trace in the output.** A call that appends a
//!   line and answers `""` every time looks exactly like one that reads and
//!   answers `""` every time, so the third append is not made.
//!
//! Changing the arguments is the only way through, and for the first two cases
//! there are no arguments to change. This is the rule's known cost. See #1301.
//!
//! ## Scope
//!
//! The ledger lives for exactly one turn (one `send_prompt` call), not for the
//! conversation. A new turn starts clean, so a call the model repeats
//! tomorrow, or in the next message, runs as usual.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// What identifies a dispatched tool call.
///
/// The name is the PROVIDER's name - the one left after `strip_location` - and
/// never the routed name the model was shown. The location root is the
/// daemon's own bookkeeping: keying on it would make the same tool on two
/// hosts two different calls, which is not what a repeat means.
///
/// The arguments are the parsed [`serde_json::Value`] re-serialized. That IS
/// the normalization the rule needs, and it is free: this workspace does not
/// enable serde_json's `preserve_order` feature, so a `serde_json::Map` is a
/// `BTreeMap` and `to_string` emits object keys in sorted order with no
/// insignificant whitespace. `{"b":2,"a":1}` and `{ "a" : 1, "b" : 2 }` are
/// therefore one key. An over-strict comparison - the raw argument string -
/// would silently do nothing, and the feature would look done.
///
/// The key holds the digest of that normalized text rather than the text, for
/// the same reason the ledger digests an output: the bytes are already in the
/// transcript, and a turn that writes a large document through a tool would
/// otherwise hold every version of it twice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RepeatKey {
    name: String,
    arguments: [u8; 32],
}

impl RepeatKey {
    pub(crate) fn new(call_name: &str, arguments: &serde_json::Value) -> Self {
        Self {
            name: call_name.to_string(),
            arguments: Sha256::digest(arguments.to_string().as_bytes()).into(),
        }
    }
}

/// What the ledger holds about one key.
struct Record {
    /// How many times the model has made this call this turn, whether it ran or
    /// was answered from the transcript. This is the number the model is told,
    /// because the number it must reason from is how often it has asked - not
    /// how often the daemon obliged.
    attempts: u32,
    /// How many times this call has actually reached a tool this turn. A
    /// suppressed call does not count: nothing ran. This is what the rule
    /// reads.
    executions: u32,
    /// The message the first result is stored under, which is what the model
    /// is told to read back. `None` until something has run.
    first_message_id: Option<String>,
    /// Digest of the first execution's output. Compared rather than kept in
    /// full: the bytes are already in the transcript, and a large result would
    /// otherwise be held twice.
    first_digest: Option<[u8; 32]>,
    /// Whether every execution so far returned the same bytes as the first.
    all_identical: bool,
}

impl Default for Record {
    fn default() -> Self {
        Self {
            attempts: 0,
            executions: 0,
            first_message_id: None,
            first_digest: None,
            // Nothing has run, so nothing has differed yet.
            all_identical: true,
        }
    }
}

/// What the loop should do with a call it is about to dispatch.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RepeatVerdict {
    /// Run it, and say nothing.
    Execute,
    /// Run it, and tell the model this exact call already ran.
    ExecuteAsRepeat { first_message_id: String },
    /// Do not run it. Answer from the transcript.
    Suppress {
        first_message_id: String,
        /// How many times the model has now made this call, including this one.
        attempts: u32,
    },
}

/// One turn's record of what it has already dispatched. See the module docs.
pub(crate) struct RepeatLedger {
    seen: HashMap<RepeatKey, Record>,
}

impl RepeatLedger {
    pub(crate) fn new() -> Self {
        Self {
            seen: HashMap::new(),
        }
    }

    /// Count one dispatch of `key`, and say what to do with it.
    ///
    /// Counting and deciding are one call because the count is part of the
    /// answer: a suppressed call never reaches [`RepeatLedger::record`], so a
    /// ledger that only counted executions would tell the tenth identical call
    /// that it was the second.
    pub(crate) fn observe_dispatch(&mut self, key: &RepeatKey) -> RepeatVerdict {
        let record = self.seen.entry(key.clone()).or_default();
        record.attempts = record.attempts.saturating_add(1);
        let Some(first_message_id) = record.first_message_id.clone() else {
            return RepeatVerdict::Execute;
        };
        if record.executions <= 1 {
            return RepeatVerdict::ExecuteAsRepeat { first_message_id };
        }
        if record.all_identical {
            RepeatVerdict::Suppress {
                first_message_id,
                attempts: record.attempts,
            }
        } else {
            RepeatVerdict::Execute
        }
    }

    /// Record one execution: which message holds its result, and what it
    /// returned.
    ///
    /// `output` is the TOOL's own output, not the message content. The repeat
    /// notice a second call carries is the daemon's own text; folding it into
    /// the comparison would make the second result differ from the first by
    /// construction, and no third call could ever be suppressed.
    pub(crate) fn record(&mut self, key: &RepeatKey, message_id: &str, output: &str) {
        let digest: [u8; 32] = Sha256::digest(output.as_bytes()).into();
        let record = self.seen.entry(key.clone()).or_default();
        record.executions = record.executions.saturating_add(1);
        match record.first_digest {
            None => {
                record.first_message_id = Some(message_id.to_string());
                record.first_digest = Some(digest);
            }
            Some(first) => record.all_identical &= first == digest,
        }
    }
}

/// The notice a second identical call carries above its own output.
///
/// Same register as the context module's `overflow_truncation_notice`: it names
/// the message the earlier bytes are stored under and the tool that reads them
/// back, so the model has somewhere to go other than a third call.
pub(crate) fn repeat_notice(first_message_id: &str) -> String {
    let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
    format!(
        "<repeat: you already made this exact call in this turn. The first \
         result is stored as message {first_message_id}. Read it with {tool} \
         message_id=\"{first_message_id}\". The output below is from this \
         second run.>"
    )
}

/// The result a suppressed call gets in place of running the tool.
///
/// Every tool call still needs a `tool_result` for provider pairing, so the
/// suppressed path pushes this one.
pub(crate) fn suppressed_notice(first_message_id: &str, attempts: u32) -> String {
    let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
    format!(
        "<not run: you have now made this exact call {attempts} times in this \
         turn. Every run of it returned the same output, so it was not run \
         again. The output is stored as message {first_message_id}. Read it \
         with {tool} message_id=\"{first_message_id}\". To get a different \
         answer, change the arguments.>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(args: &str) -> RepeatKey {
        RepeatKey::new(
            "probe",
            &serde_json::from_str::<serde_json::Value>(args).unwrap(),
        )
    }

    #[test]
    fn reordered_keys_and_whitespace_produce_one_key() {
        assert_eq!(key(r#"{"a":1,"b":2}"#), key(r#"{"b":2,"a":1}"#));
        assert_eq!(key(r#"{"a":1,"b":2}"#), key("{ \"a\" : 1 ,\n \"b\" : 2 }"));
    }

    #[test]
    fn different_arguments_and_different_names_are_different_keys() {
        assert_ne!(key(r#"{"a":1}"#), key(r#"{"a":2}"#));
        let other = RepeatKey::new("other", &serde_json::json!({"a": 1}));
        assert_ne!(key(r#"{"a":1}"#), other);
    }

    #[test]
    fn nested_object_keys_are_normalized_too() {
        let a = RepeatKey::new("probe", &serde_json::json!({"o": {"x": 1, "y": 2}}));
        let b = RepeatKey::new("probe", &serde_json::json!({"o": {"y": 2, "x": 1}}));
        assert_eq!(a, b);
    }

    #[test]
    fn array_order_is_significant() {
        // Normalization must not reach into value semantics: [1,2] and [2,1]
        // are different arguments, and a tool may well answer differently.
        let a = RepeatKey::new("probe", &serde_json::json!({"xs": [1, 2]}));
        let b = RepeatKey::new("probe", &serde_json::json!({"xs": [2, 1]}));
        assert_ne!(a, b);
    }

    /// Drive one call the way the turn loop does: ask, then record what ran.
    fn dispatch(
        ledger: &mut RepeatLedger,
        args: &str,
        message_id: &str,
        output: &str,
    ) -> RepeatVerdict {
        let k = key(args);
        let verdict = ledger.observe_dispatch(&k);
        if !matches!(verdict, RepeatVerdict::Suppress { .. }) {
            ledger.record(&k, message_id, output);
        }
        verdict
    }

    #[test]
    fn the_first_call_of_a_turn_executes_unannounced() {
        let mut ledger = RepeatLedger::new();
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", "same"),
            RepeatVerdict::Execute
        );
    }

    #[test]
    fn the_second_call_executes_and_names_the_first_result() {
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", "same");
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", "same"),
            RepeatVerdict::ExecuteAsRepeat {
                first_message_id: "msg-1".to_string()
            }
        );
    }

    #[test]
    fn the_third_call_is_suppressed_when_both_runs_returned_the_same_bytes() {
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", "same");
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", "same");
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-3", "same"),
            RepeatVerdict::Suppress {
                first_message_id: "msg-1".to_string(),
                attempts: 3,
            }
        );
    }

    #[test]
    fn a_suppressed_call_still_counts_toward_what_the_model_is_told() {
        // A suppressed call never reaches `record`, so a ledger that counted
        // only executions would tell the tenth identical call it was the
        // second. The number the model reasons from is how often it has asked.
        let mut ledger = RepeatLedger::new();
        for i in 1..=3 {
            dispatch(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), "same");
        }
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-4", "same"),
            RepeatVerdict::Suppress {
                first_message_id: "msg-1".to_string(),
                attempts: 4,
            }
        );
    }

    #[test]
    fn a_call_whose_output_changed_on_its_first_repeat_keeps_executing() {
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", "running");
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", "done");
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-3", "done"),
            RepeatVerdict::Execute
        );
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-4", "done"),
            RepeatVerdict::Execute,
            "one changed output makes the tool time-varying for the rest of the turn"
        );
    }

    #[test]
    fn two_matching_runs_freeze_the_key_even_though_the_answer_would_change() {
        // The rule's known cost, pinned so it cannot change by accident. Two
        // identical runs suppress every later call, and a suppressed call
        // records nothing, so no later answer can reopen the key. #1301 holds
        // the case this loses.
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", "running");
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", "running");
        assert!(matches!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-3", "done"),
            RepeatVerdict::Suppress { .. }
        ));
        assert!(matches!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-4", "done"),
            RepeatVerdict::Suppress { .. }
        ));
    }

    #[test]
    fn one_key_does_not_answer_for_another() {
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", "same");
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", "same");
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":2}"#, "msg-3", "same"),
            RepeatVerdict::Execute
        );
    }

    #[test]
    fn both_notices_name_the_message_and_the_readback_tool() {
        let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
        let repeat = repeat_notice("msg-1");
        assert!(
            repeat.contains("msg-1") && repeat.contains(tool),
            "{repeat}"
        );
        let skipped = suppressed_notice("msg-1", 3);
        assert!(
            skipped.contains("msg-1") && skipped.contains(tool),
            "{skipped}"
        );
    }
}
