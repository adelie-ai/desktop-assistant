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
//! - A poll whose value changes is never suppressed. A poll that returns the
//!   same bytes twice has reported no change, and the third read of no change
//!   is answered from the transcript.
//! - Re-reading a file after writing it is never suppressed, because the write
//!   changed the bytes and the second read differs from the first.
//!
//! What it does not distinguish is a side effect that leaves no trace in the
//! output. A call that appends a line and answers `""` every time looks
//! identical to a call that reads and answers `""` every time, so the third
//! append is not made. Changing the arguments - which the notice asks for -
//! is the way through.
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RepeatKey {
    name: String,
    arguments: String,
}

impl RepeatKey {
    pub(crate) fn new(call_name: &str, arguments: &serde_json::Value) -> Self {
        Self {
            name: call_name.to_string(),
            arguments: arguments.to_string(),
        }
    }
}

/// What the ledger holds about one key.
struct Record {
    /// How many times this call has actually reached a tool this turn. A
    /// suppressed call does not count: nothing ran.
    executions: u32,
    /// The message the first result is stored under, which is what the model
    /// is told to read back.
    first_message_id: String,
    /// Digest of the first execution's output. Compared rather than kept in
    /// full: the bytes are already in the transcript, and a large result would
    /// otherwise be held twice.
    first_digest: [u8; 32],
    /// Whether every execution so far returned the same bytes as the first.
    all_identical: bool,
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
        executions: u32,
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

    /// What to do with `key`, given what this turn has already run.
    pub(crate) fn verdict(&self, key: &RepeatKey) -> RepeatVerdict {
        let Some(record) = self.seen.get(key) else {
            return RepeatVerdict::Execute;
        };
        if record.executions <= 1 {
            return RepeatVerdict::ExecuteAsRepeat {
                first_message_id: record.first_message_id.clone(),
            };
        }
        if record.all_identical {
            RepeatVerdict::Suppress {
                first_message_id: record.first_message_id.clone(),
                executions: record.executions,
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
    pub(crate) fn record(&mut self, key: RepeatKey, message_id: &str, output: &str) {
        let digest: [u8; 32] = Sha256::digest(output.as_bytes()).into();
        self.seen
            .entry(key)
            .and_modify(|record| {
                record.executions = record.executions.saturating_add(1);
                record.all_identical &= record.first_digest == digest;
            })
            .or_insert(Record {
                executions: 1,
                first_message_id: message_id.to_string(),
                first_digest: digest,
                all_identical: true,
            });
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
pub(crate) fn suppressed_notice(first_message_id: &str, executions: u32) -> String {
    let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
    format!(
        "<not run: you already made this exact call {executions} times in this \
         turn, and every run returned the same output, so it was not run \
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

    #[test]
    fn the_first_call_of_a_turn_executes_unannounced() {
        let ledger = RepeatLedger::new();
        assert_eq!(ledger.verdict(&key(r#"{"a":1}"#)), RepeatVerdict::Execute);
    }

    #[test]
    fn the_second_call_executes_and_names_the_first_result() {
        let mut ledger = RepeatLedger::new();
        ledger.record(key(r#"{"a":1}"#), "msg-1", "same");
        assert_eq!(
            ledger.verdict(&key(r#"{"a":1}"#)),
            RepeatVerdict::ExecuteAsRepeat {
                first_message_id: "msg-1".to_string()
            }
        );
    }

    #[test]
    fn the_third_call_is_suppressed_when_both_runs_returned_the_same_bytes() {
        let mut ledger = RepeatLedger::new();
        ledger.record(key(r#"{"a":1}"#), "msg-1", "same");
        ledger.record(key(r#"{"a":1}"#), "msg-2", "same");
        assert_eq!(
            ledger.verdict(&key(r#"{"a":1}"#)),
            RepeatVerdict::Suppress {
                first_message_id: "msg-1".to_string(),
                executions: 2,
            }
        );
    }

    #[test]
    fn a_call_whose_output_changed_keeps_executing() {
        let mut ledger = RepeatLedger::new();
        ledger.record(key(r#"{"a":1}"#), "msg-1", "running");
        ledger.record(key(r#"{"a":1}"#), "msg-2", "done");
        assert_eq!(ledger.verdict(&key(r#"{"a":1}"#)), RepeatVerdict::Execute);
        ledger.record(key(r#"{"a":1}"#), "msg-3", "done");
        assert_eq!(
            ledger.verdict(&key(r#"{"a":1}"#)),
            RepeatVerdict::Execute,
            "one changed output makes the tool time-varying for the rest of the turn"
        );
    }

    #[test]
    fn one_key_does_not_answer_for_another() {
        let mut ledger = RepeatLedger::new();
        ledger.record(key(r#"{"a":1}"#), "msg-1", "same");
        ledger.record(key(r#"{"a":1}"#), "msg-2", "same");
        assert_eq!(ledger.verdict(&key(r#"{"a":2}"#)), RepeatVerdict::Execute);
    }

    #[test]
    fn both_notices_name_the_message_and_the_readback_tool() {
        let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
        let repeat = repeat_notice("msg-1");
        assert!(
            repeat.contains("msg-1") && repeat.contains(tool),
            "{repeat}"
        );
        let skipped = suppressed_notice("msg-1", 2);
        assert!(
            skipped.contains("msg-1") && skipped.contains(tool),
            "{skipped}"
        );
    }
}
