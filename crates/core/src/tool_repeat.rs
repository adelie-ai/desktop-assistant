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
//! ## Two separate answers, and only one of them withholds anything
//!
//! **A result that repeats the one before it is not appended again.** When a
//! call runs and returns exactly what that key returned on its previous run,
//! the turn appends a pointer to the message already holding those bytes
//! instead of a second copy - provided the pointer is smaller than the bytes.
//! The tool ran, so nothing here can be stale, and this is what actually breaks
//! the fetch/evict/refetch loop.
//!
//! It compares against the previous result and not against every result the key
//! has ever produced, so a tool that alternates between two answers - A, B, A,
//! B - stores both of them every time and saves nothing. Nothing is wrong in
//! that case; there is simply nothing here for it.
//!
//! **Suppression is an execution saving on top of that.** Two matching runs of
//! at least [`MIN_SUPPRESSIBLE_BYTES`] make a key suppressible; from there the
//! loop answers some calls from the transcript without running the tool. That
//! one CAN be stale, so it is bounded - see below - and it says so in the
//! result the model reads. A suppressed call still costs its round, so what it
//! saves is the execution, which is why a result too small to be worth
//! replacing is never suppressed at all.
//!
//! ## The backoff, and why suppression must never be terminal
//!
//! An earlier rule suppressed every call of a suppressible key. That froze the
//! key: a suppressed call does not execute, so it records nothing, so nothing
//! could ever show that the answer had changed. A subagent poll that read
//! `running` twice never saw the child complete, and a file read twice before a
//! write returned the pre-write bytes for the rest of the turn - which is the
//! case the rule existed to protect.
//!
//! So each key carries a suppression counter and a threshold that starts at
//! [`INITIAL_THRESHOLD`]:
//!
//! - Each suppressed call increments the counter.
//! - When the counter reaches the threshold the call RUNS, the counter resets
//!   to zero, and the threshold doubles.
//! - Any run whose result differs from the previous one clears the suppressible
//!   state outright, and the key is back to needing two matching runs.
//!
//! Over 21 identical calls the tool runs about five times rather than 21, and
//! no key can freeze. A value that changes is seen a bounded number of rounds
//! late, and that bound only grows when the tool has been re-run in between.
//!
//! ## What the rule still does not hold
//!
//! A side effect that leaves no trace in the output. A call that appends a line
//! and answers `""` looks exactly like one that reads and answers `""`. The
//! floor covers the worst of this by accident and then on purpose: an empty
//! success renders to a 49-byte marker, far under
//! [`MIN_SUPPRESSIBLE_BYTES`], so a side effect that says nothing is never
//! suppressible at all. What remains is a call that changes something AND
//! returns half a kilobyte of unchanging text, where the text is itself
//! evidence that nothing observable moved. The backoff bounds that too.
//!
//! An error is recorded like any other output, so a server that answers
//! identically twice while it restarts is suppressed for a bounded run of calls
//! even after the cause is repaired. The backoff is what makes that recoverable
//! rather than permanent.
//!
//! ## Scope
//!
//! The ledger lives for exactly one turn (one `send_prompt` call), not for the
//! conversation. A new turn starts clean, so a call the model repeats
//! tomorrow, or in the next message, runs as usual.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::tool_routing::ToolConnection;

/// Suppressions a suppressible key may take before the next call runs anyway.
///
/// Two, because two is what it took to decide the key was repeating itself: the
/// evidence and the bound are the same size, so the first thing the rule does
/// after concluding "this is not changing" is to go and check.
const INITIAL_THRESHOLD: u32 = 2;

/// The largest the threshold may grow to.
///
/// Doubling without a ceiling makes the property "no key can freeze" true only
/// in the limit, and a turn is not the limit: unbounded, the runs land on calls
/// 1, 2, 5, 10, 19, 36, 69, 134, 263, so a key asked for 134 times is not
/// re-checked before `MAX_TOOL_ROUNDS` ends the turn. A child that finished at
/// round 80 would go unnoticed to the end. Capped, a key called on every round
/// of a full turn is re-checked at least ten times - no stretch exceeding a
/// tenth of the round budget, the tail after the last run included - and the
/// first several fires are unchanged: 21 identical calls still run the tool
/// five times.
///
/// Nineteen is the largest value that still holds it, and the test that states
/// it goes red above that - so this is a choice with margin rather than the
/// edge of one.
const MAX_THRESHOLD: u32 = 16;

/// The smallest result this rule will answer from the transcript.
///
/// A tool answered from the transcript still costs its round; what it saves is
/// the execution and the bytes. Below this size it saves neither. Both notices
/// render to under 512 bytes with a UUIDv7 message id, so a shorter result
/// replaced by one makes the context BIGGER - which is the one thing a rule
/// that exists to stop the context refilling may not do.
///
/// The same line is held either side of this module: `planning`'s
/// `COMPACTION_MIN_EVICT_BYTES` is 512 for the same reason, and its eviction
/// refuses any pointer no smaller than what it replaces. Small results are also
/// where a stale answer costs most - a poll's `{"status":"running"}` is a few
/// dozen bytes - so leaving them alone is the safe direction twice over.
const MIN_SUPPRESSIBLE_BYTES: usize = 512;

/// The digest of one tool result. Computed once per execution and passed
/// between the ledger's calls, so a multi-megabyte payload is hashed once.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResultDigest([u8; 32]);

impl ResultDigest {
    pub(crate) fn of(output: &str) -> Self {
        Self(Sha256::digest(output.as_bytes()).into())
    }
}

/// What identifies a dispatched tool call.
///
/// ## The name carries its location, unlike a burn identity
///
/// A burn - the negative-memory key at `crate::service::burn_identity` - is
/// deliberately keyed on the PROVIDER name with the location root stripped,
/// because a lesson about a tool should be portable: "this went badly" is worth
/// knowing about the same tool on another machine.
///
/// This key asks the opposite question. It does not ask what a tool is like; it
/// asks whether THIS call was already made. Reading a path on the daemon tells
/// you nothing about the same path on the user's own machine, so the same
/// provider tool on two connections is two calls. Merging them can serve one
/// host's bytes as the other's, which is a wrong answer rather than waste.
///
/// Keep the two keyed differently. The contrast is deliberate and reads as an
/// inconsistency to anyone who meets one without the other.
///
/// ## The arguments
///
/// The parsed [`serde_json::Value`] re-serialized. That IS the normalization
/// the rule needs, and it is free: this workspace does not enable serde_json's
/// `preserve_order` feature, so a `serde_json::Map` is a `BTreeMap` and
/// `to_string` emits object keys in sorted order with no insignificant
/// whitespace. `{"b":2,"a":1}` and `{ "a" : 1, "b" : 2 }` are therefore one
/// key. An over-strict comparison - the raw argument string - would silently do
/// nothing, and the feature would look done.
///
/// The key holds the digest of that normalized text rather than the text, for
/// the same reason the ledger digests an output: the bytes are already in the
/// transcript, and a turn that writes a large document through a tool would
/// otherwise hold every version of it twice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RepeatKey {
    /// The connection that runs it, by its own label. `None` for a name the
    /// round's table does not hold - the model calling a tool it learned in an
    /// earlier turn - which the daemon's executor runs. Two such calls of one
    /// name are one key, because one executor answers both.
    location: Option<String>,
    name: String,
    arguments: [u8; 32],
}

impl RepeatKey {
    pub(crate) fn new(
        connection: Option<&ToolConnection>,
        call_name: &str,
        arguments: &serde_json::Value,
    ) -> Self {
        Self {
            location: connection.map(ToolConnection::label),
            name: call_name.to_string(),
            arguments: Sha256::digest(arguments.to_string().as_bytes()).into(),
        }
    }
}

/// Where a key's current bytes are stored, and what they are.
///
/// One value rather than two fields, so the half-built state cannot be written:
/// an id without a digest would make every later comparison take the "nothing
/// to compare with" arm, and a tool whose answer changes would read as one that
/// repeats itself.
struct Held {
    /// The message holding these bytes, which is what the model is told to read
    /// back. A run that reproduces them appends a pointer, so this keeps naming
    /// the message that actually carries them.
    message_id: String,
    digest: ResultDigest,
}

/// What the ledger holds about one key.
struct Record {
    /// How many dispatches of this call have reached the ledger this turn,
    /// whether they ran or were answered from the transcript. This is the
    /// number the model is told, because what it must reason from is how often
    /// it has asked - not how often the daemon obliged.
    ///
    /// A call refused before dispatch - malformed argument JSON, a burn hold, a
    /// named-only call missing a required argument - never reaches here and is
    /// not counted.
    attempts: u32,
    /// The most recent distinct result, and where it is stored.
    held: Option<Held>,
    /// Whether two runs in a row have returned the same bytes. Cleared outright
    /// by a run that returns something else.
    suppressible: bool,
    /// Suppressions since the last run. Reaching `threshold` runs the tool.
    suppressions: u32,
    /// How many suppressions this key may take before the next call runs.
    /// Doubles each time it fires, so a key the model keeps asking for costs
    /// logarithmically many runs rather than all of them or none.
    threshold: u32,
}

impl Default for Record {
    fn default() -> Self {
        Self {
            attempts: 0,
            held: None,
            suppressible: false,
            suppressions: 0,
            threshold: INITIAL_THRESHOLD,
        }
    }
}

/// What the loop should do with a call it is about to dispatch.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RepeatVerdict {
    /// Run it.
    Execute,
    /// Do not run it. Answer from the transcript, and say that is what
    /// happened.
    Suppress {
        /// The message holding the bytes this call would most likely have
        /// returned. They are from an earlier run, so they may be stale.
        message_id: String,
        /// How many times the model has now made this call, including this one.
        attempts: u32,
    },
}

/// What to do with the bytes a run just produced.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ResultDisposition {
    /// Nothing holds these bytes. Store them.
    Store,
    /// This message already holds exactly these bytes. Point at it.
    SameAs { message_id: String },
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

    /// Count one dispatch of `key`, and say whether to run it.
    ///
    /// `may_suppress` is false for a call the loop must always run whatever the
    /// ledger concludes - see the exemption at the call site. Such a call is
    /// still counted, and its result still becomes a pointer when it repeats,
    /// but it never spends the backoff.
    ///
    /// Counting and deciding are one call because the count is part of the
    /// answer: a suppressed call never reaches [`RepeatLedger::record`], so a
    /// ledger that only counted runs would tell the tenth identical call that
    /// it was the second.
    pub(crate) fn observe_dispatch(
        &mut self,
        key: &RepeatKey,
        may_suppress: bool,
    ) -> RepeatVerdict {
        let record = self.seen.entry(key.clone()).or_default();
        record.attempts = record.attempts.saturating_add(1);
        if !may_suppress || !record.suppressible {
            return RepeatVerdict::Execute;
        }
        // `suppressible` is only ever set beside a `held`, so this cannot be
        // the "nothing to point at" case - but ask rather than assume, because
        // the alternative to an answer here is a suppression with nowhere to
        // send the model.
        let Some(held) = record.held.as_ref() else {
            return RepeatVerdict::Execute;
        };
        if record.suppressions >= record.threshold {
            // The bound is up. Run it, start the count again, and give the key
            // twice as long before the next check.
            record.suppressions = 0;
            record.threshold = record.threshold.saturating_mul(2).min(MAX_THRESHOLD);
            return RepeatVerdict::Execute;
        }
        let message_id = held.message_id.clone();
        record.suppressions = record.suppressions.saturating_add(1);
        RepeatVerdict::Suppress {
            message_id,
            attempts: record.attempts,
        }
    }

    /// Whether the bytes a run produced are already in the transcript.
    ///
    /// Asked before the result message is built, because the answer decides
    /// what that message carries.
    ///
    /// Held to the same floor as suppression, and not merely to "is the pointer
    /// shorter". A pointer is 307 bytes, so a size comparison alone starts
    /// pointing at 308: a 400-byte result becomes a 307-byte address, saving 93
    /// bytes, and if the model spends a round on
    /// [`crate::ports::transcript::TRANSCRIPT_GET_TOOL`] to get it back the turn
    /// pays an LLM call and some 460 bytes to save those 93. The trade is only
    /// worth making where the bytes dwarf the address.
    pub(crate) fn disposition(
        &self,
        key: &RepeatKey,
        digest: ResultDigest,
        output_bytes: usize,
    ) -> ResultDisposition {
        if output_bytes < MIN_SUPPRESSIBLE_BYTES {
            return ResultDisposition::Store;
        }
        match self.seen.get(key).and_then(|record| record.held.as_ref()) {
            Some(held) if held.digest == digest => ResultDisposition::SameAs {
                message_id: held.message_id.clone(),
            },
            _ => ResultDisposition::Store,
        }
    }

    /// Record one run: which message the turn appended for it, and what the
    /// tool returned.
    ///
    /// `digest` is of the TOOL's own output, never of the message content. A
    /// message carrying a pointer is shorter than the bytes it names, and
    /// digesting that instead would make every repeat look like a change.
    pub(crate) fn record(
        &mut self,
        key: &RepeatKey,
        message_id: &str,
        digest: ResultDigest,
        output_bytes: usize,
    ) {
        let record = self.seen.entry(key.clone()).or_default();
        match &record.held {
            None => {
                record.held = Some(Held {
                    message_id: message_id.to_string(),
                    digest,
                });
            }
            Some(held) if held.digest == digest => {
                // The same bytes, so the message just appended points at them
                // and `held` must keep naming the message that carries them.
                // Two runs in a row agreeing is what makes the key
                // suppressible - but only above the size where answering from
                // the transcript is cheaper than answering with the bytes.
                record.suppressible = output_bytes >= MIN_SUPPRESSIBLE_BYTES;
            }
            Some(_) => {
                // Something changed. The key is not repeating itself after all,
                // so it goes back to the plain rule and the backoff starts over.
                record.held = Some(Held {
                    message_id: message_id.to_string(),
                    digest,
                });
                record.suppressible = false;
                record.suppressions = 0;
                record.threshold = INITIAL_THRESHOLD;
            }
        }
    }
}

/// The result of a run that returned exactly what an earlier run returned.
///
/// It must not read as a refusal: the tool RAN, and these are this run's own
/// bytes. Only [`suppressed_notice`] describes a call that did not happen, and
/// the two are worded so a model can tell fresh from stale at a glance.
pub(crate) fn same_bytes_notice(message_id: &str) -> String {
    let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
    format!(
        "<same as before: this call ran, and it returned exactly the bytes it \
         returned earlier in this turn, so they are not repeated here. They are \
         stored as message {message_id}, and they are current. Read them with \
         {tool} message_id=\"{message_id}\".>"
    )
}

/// The result a suppressed call gets in place of running the tool.
///
/// Says outright that the tool did not run, because this is the one result the
/// model holds that may be out of date, and a model that cannot tell it from a
/// fresh answer will act on it believing it is current.
///
/// Every tool call still needs a `tool_result` for provider pairing, so the
/// suppressed path pushes this one.
pub(crate) fn suppressed_notice(message_id: &str, attempts: u32) -> String {
    let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
    format!(
        "<not run: you have now made this exact call {attempts} times in this \
         turn, and the last runs all returned the same output. The tool did \
         not run this time, so what follows is not a fresh answer: it is the \
         result of an earlier run, stored as message {message_id}, and it may \
         be out of date. Read it with {tool} message_id=\"{message_id}\". This \
         call runs again on its own after a few more attempts; to get a fresh \
         answer now, change the arguments.>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(args: &str) -> RepeatKey {
        RepeatKey::new(
            None,
            "probe",
            &serde_json::from_str::<serde_json::Value>(args).unwrap(),
        )
    }

    /// A result the rule applies to: at or above the size where answering from
    /// the transcript is cheaper than answering with the bytes. `label`
    /// distinguishes one from another.
    fn big(label: &str) -> String {
        format!("{label}{}", "x".repeat(MIN_SUPPRESSIBLE_BYTES))
    }

    /// Drive one call the way the turn loop does: ask, run it if allowed, then
    /// record what it returned. Returns what the model would be handed.
    fn dispatch(ledger: &mut RepeatLedger, args: &str, message_id: &str, output: &str) -> String {
        let k = key(args);
        match ledger.observe_dispatch(&k, true) {
            RepeatVerdict::Suppress {
                message_id,
                attempts,
            } => suppressed_notice(&message_id, attempts),
            RepeatVerdict::Execute => {
                let digest = ResultDigest::of(output);
                let content = match ledger.disposition(&k, digest, output.len()) {
                    ResultDisposition::SameAs { message_id } => same_bytes_notice(&message_id),
                    ResultDisposition::Store => output.to_string(),
                };
                ledger.record(&k, message_id, digest, output.len());
                content
            }
        }
    }

    fn ran(ledger: &mut RepeatLedger, args: &str, message_id: &str, output: &str) -> bool {
        !dispatch(ledger, args, message_id, output).starts_with("<not run:")
    }

    #[test]
    fn reordered_keys_and_whitespace_produce_one_key() {
        assert_eq!(key(r#"{"a":1,"b":2}"#), key(r#"{"b":2,"a":1}"#));
        assert_eq!(key(r#"{"a":1,"b":2}"#), key("{ \"a\" : 1 ,\n \"b\" : 2 }"));
    }

    #[test]
    fn different_arguments_and_different_names_are_different_keys() {
        assert_ne!(key(r#"{"a":1}"#), key(r#"{"a":2}"#));
        let other = RepeatKey::new(None, "other", &serde_json::json!({"a": 1}));
        assert_ne!(key(r#"{"a":1}"#), other);
    }

    #[test]
    fn one_name_on_two_connections_is_two_keys() {
        let args = serde_json::json!({"path": "/etc/hosts"});
        let daemon = RepeatKey::new(Some(&ToolConnection::daemon_builtins()), "read_file", &args);
        let client = RepeatKey::new(Some(&ToolConnection::client_device()), "read_file", &args);
        assert_ne!(daemon, client);
        // And two servers on the daemon are two connections as well.
        let one = RepeatKey::new(
            Some(&ToolConnection::daemon_server("fileio")),
            "read",
            &args,
        );
        let two = RepeatKey::new(Some(&ToolConnection::daemon_server("vault")), "read", &args);
        assert_ne!(one, two);
    }

    #[test]
    fn nested_object_keys_are_normalized_too() {
        let a = RepeatKey::new(None, "probe", &serde_json::json!({"o": {"x": 1, "y": 2}}));
        let b = RepeatKey::new(None, "probe", &serde_json::json!({"o": {"y": 2, "x": 1}}));
        assert_eq!(a, b);
    }

    #[test]
    fn array_order_is_significant() {
        // Normalization must not reach into value semantics: [1,2] and [2,1]
        // are different arguments, and a tool may well answer differently.
        let a = RepeatKey::new(None, "probe", &serde_json::json!({"xs": [1, 2]}));
        let b = RepeatKey::new(None, "probe", &serde_json::json!({"xs": [2, 1]}));
        assert_ne!(a, b);
    }

    #[test]
    fn the_first_two_calls_both_run() {
        let mut ledger = RepeatLedger::new();
        assert!(ran(&mut ledger, r#"{"a":1}"#, "msg-1", &big("same")));
        assert!(ran(&mut ledger, r#"{"a":1}"#, "msg-2", &big("same")));
    }

    #[test]
    fn a_run_that_reproduces_earlier_bytes_points_at_them_instead() {
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", &big("same"));
        let second = dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", &big("same"));
        assert!(second.contains("msg-1"), "{second}");
        assert!(
            !second.contains("did not run"),
            "the tool ran; the result must not say otherwise: {second}"
        );
    }

    #[test]
    fn the_suppression_threshold_doubles_each_time_it_fires() {
        // Twenty-one identical calls run the tool on 1, 2, 5, 10 and 19: two to
        // decide it repeats, then a check after 2, 4 and 8 suppressions.
        let mut ledger = RepeatLedger::new();
        let ran_on: Vec<usize> = (1..=21)
            .filter(|i| ran(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same")))
            .collect();
        assert_eq!(ran_on, vec![1, 2, 5, 10, 19]);
    }

    #[test]
    fn a_result_too_small_to_be_worth_replacing_is_never_suppressed() {
        // Both notices are hundreds of bytes. A twenty-byte poll answered with
        // one costs more context than running the tool, which inverts the whole
        // point - so below the floor the rule stands aside.
        let mut ledger = RepeatLedger::new();
        let small = r#"{"status":"running"}"#;
        assert!(small.len() < MIN_SUPPRESSIBLE_BYTES);
        for i in 1..=20 {
            assert!(
                ran(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), small),
                "call {i} must run"
            );
        }
    }

    #[test]
    fn a_small_repeated_result_is_stored_rather_than_replaced_by_a_longer_pointer() {
        let mut ledger = RepeatLedger::new();
        let small = "ok";
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", small);
        let second = dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", small);
        assert_eq!(
            second, small,
            "a pointer longer than the bytes it names must not replace them"
        );
    }

    #[test]
    fn a_result_between_the_pointer_length_and_the_floor_is_still_stored() {
        // The band a "is the pointer shorter" test would have taken: 400 bytes
        // replaced by a 307-byte address saves 93 and risks a whole round to
        // read them back.
        let mut ledger = RepeatLedger::new();
        let middling = "x".repeat(400);
        assert!(middling.len() > same_bytes_notice("msg-1").len());
        assert!(middling.len() < MIN_SUPPRESSIBLE_BYTES);
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", &middling);
        assert_eq!(
            dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", &middling),
            middling
        );
    }

    #[test]
    fn a_key_called_every_round_of_a_turn_is_re_checked_at_least_ten_times() {
        // The ceiling's claim is about a TURN, so measure it against the turn.
        //
        // Two traps here, and the second is why the obvious test is useless.
        // Asserting the gap against `MAX_THRESHOLD` is circular - it passes for
        // any ceiling, including one that never fires. And measuring only the
        // gaps BETWEEN runs misses the tail: with the ceiling raised to a
        // million the runs land on calls 1, 2, 5, 10, 19, 36, 69, 134 and the
        // ninth would be call 263, so the widest gap between runs is 65 while
        // the last 66 calls of the turn get no check at all. The turn ending is
        // the widest gap of all, and it is the one that matters.
        let mut ledger = RepeatLedger::new();
        let ran_on: Vec<usize> = (1..=crate::service::MAX_TOOL_ROUNDS)
            .filter(|i| ran(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same")))
            .collect();
        let last = *ran_on.last().expect("the tool ran at least once");
        let widest = ran_on
            .windows(2)
            .map(|w| w[1] - w[0])
            .chain(std::iter::once(crate::service::MAX_TOOL_ROUNDS - last))
            .max()
            .expect("at least one gap");
        assert!(
            widest * 10 <= crate::service::MAX_TOOL_ROUNDS,
            "a key called every round must be re-checked at least ten times in \
             a {}-round turn, so no stretch may exceed a tenth of it; the \
             widest was {widest} (ran on {ran_on:?})",
            crate::service::MAX_TOOL_ROUNDS
        );
    }

    #[test]
    fn no_key_can_freeze_however_many_times_it_is_called() {
        // The defect this rule replaced. Whatever the count, the next run is
        // always a bounded number of calls away.
        let mut ledger = RepeatLedger::new();
        for i in 1..=200 {
            dispatch(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same"));
        }
        let further: Vec<usize> = (201..=600)
            .filter(|i| ran(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same")))
            .collect();
        assert!(
            !further.is_empty(),
            "a key must never stop being re-checked"
        );
    }

    #[test]
    fn a_value_that_changes_while_suppressed_reaches_the_model_when_the_bound_fires() {
        // The case the terminal rule lost: two identical polls, then a value
        // the model must see.
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-1", &big("running"));
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-2", &big("running"));
        assert!(!ran(&mut ledger, r#"{"a":1}"#, "msg-3", &big("running")));
        assert!(!ran(&mut ledger, r#"{"a":1}"#, "msg-4", &big("running")));
        let fifth = dispatch(&mut ledger, r#"{"a":1}"#, "msg-5", &big("completed"));
        assert_eq!(
            fifth,
            big("completed"),
            "the bound must fire and hand the model the new value"
        );
    }

    #[test]
    fn a_pointer_names_the_message_that_actually_holds_those_bytes() {
        // A, then B, then B again. The pointer the third run gets must name the
        // message carrying B - not the one carrying A, which `held` named until
        // the change. Getting this wrong tells the model bytes are "stored as
        // message A, and they are current" when message A holds something else,
        // which is the one thing this feature must never do.
        let mut ledger = RepeatLedger::new();
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-a", &big("A"));
        dispatch(&mut ledger, r#"{"a":1}"#, "msg-b", &big("B"));
        let third = dispatch(&mut ledger, r#"{"a":1}"#, "msg-c", &big("B"));
        assert!(third.contains("msg-b"), "{third}");
        assert!(!third.contains("msg-a"), "{third}");
    }

    #[test]
    fn a_changed_result_clears_the_suppressible_state_outright() {
        // Back to the plain rule: it takes two matching runs again, so the very
        // next call cannot be suppressed however long the key was repeating.
        let mut ledger = RepeatLedger::new();
        for i in 1..=4 {
            dispatch(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same"));
        }
        assert!(ran(&mut ledger, r#"{"a":1}"#, "msg-5", &big("changed")));
        assert!(ran(
            &mut ledger,
            r#"{"a":1}"#,
            "msg-6",
            &big("changed again")
        ));
        assert!(ran(
            &mut ledger,
            r#"{"a":1}"#,
            "msg-7",
            &big("changed once more")
        ));
    }

    #[test]
    fn a_call_the_loop_must_always_make_is_never_suppressed() {
        let mut ledger = RepeatLedger::new();
        let k = key(r#"{"a":1}"#);
        for i in 1..=20 {
            assert_eq!(
                ledger.observe_dispatch(&k, false),
                RepeatVerdict::Execute,
                "call {i} must run"
            );
            let output = big("same");
            let digest = ResultDigest::of(&output);
            ledger.record(&k, &format!("msg-{i}"), digest, output.len());
        }
    }

    #[test]
    fn an_exempt_call_still_points_at_bytes_the_transcript_already_holds() {
        // Exemption gives up the execution saving, not the context saving.
        let mut ledger = RepeatLedger::new();
        let k = key(r#"{"a":1}"#);
        let output = big("same");
        let digest = ResultDigest::of(&output);
        ledger.observe_dispatch(&k, false);
        ledger.record(&k, "msg-1", digest, output.len());
        ledger.observe_dispatch(&k, false);
        assert_eq!(
            ledger.disposition(&k, digest, output.len()),
            ResultDisposition::SameAs {
                message_id: "msg-1".to_string()
            }
        );
    }

    #[test]
    fn one_key_does_not_answer_for_another() {
        let mut ledger = RepeatLedger::new();
        for i in 1..=4 {
            dispatch(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same"));
        }
        assert!(ran(&mut ledger, r#"{"a":2}"#, "msg-9", &big("same")));
    }

    #[test]
    fn a_suppressed_call_still_counts_toward_what_the_model_is_told() {
        // A suppressed call never reaches `record`, so a ledger that counted
        // only runs would tell the fourth identical call it was the second.
        let mut ledger = RepeatLedger::new();
        for i in 1..=3 {
            dispatch(&mut ledger, r#"{"a":1}"#, &format!("msg-{i}"), &big("same"));
        }
        let fourth = dispatch(&mut ledger, r#"{"a":1}"#, "msg-4", &big("same"));
        assert!(fourth.contains("4 times"), "{fourth}");
    }

    #[test]
    fn both_notices_name_the_message_and_the_readback_tool() {
        let tool = crate::ports::transcript::TRANSCRIPT_GET_TOOL;
        let same = same_bytes_notice("msg-1");
        assert!(same.contains("msg-1") && same.contains(tool), "{same}");
        let skipped = suppressed_notice("msg-1", 3);
        assert!(
            skipped.contains("msg-1") && skipped.contains(tool),
            "{skipped}"
        );
    }

    #[test]
    fn only_the_suppressed_notice_says_the_tool_did_not_run() {
        assert!(suppressed_notice("msg-1", 3).contains("did not run"));
        assert!(!same_bytes_notice("msg-1").contains("did not run"));
    }

    #[test]
    fn a_suppressed_result_warns_that_it_may_be_out_of_date() {
        // "The tool did not run" alone leaves the model to infer the
        // consequence. This is the one result it holds that may be wrong, so
        // the warning is part of the contract and not decoration.
        let skipped = suppressed_notice("msg-1", 3);
        assert!(skipped.contains("out of date"), "{skipped}");
        assert!(skipped.contains("not a fresh answer"), "{skipped}");
    }

    #[test]
    fn a_pointer_from_a_run_says_the_call_ran_and_the_bytes_are_current() {
        // The other half of the same contract: not saying "did not run" is not
        // the same as saying it did.
        let same = same_bytes_notice("msg-1");
        assert!(same.contains("this call ran"), "{same}");
        assert!(same.contains("current"), "{same}");
    }

    #[test]
    fn the_suppressed_notice_carries_the_attempt_count_it_is_given() {
        // The count is the whole point of counting attempts, and the ledger
        // holding the right number buys nothing if the sentence drops it.
        assert!(suppressed_notice("msg-1", 7).contains('7'));
        assert!(suppressed_notice("msg-1", 12).contains("12"));
    }
}
