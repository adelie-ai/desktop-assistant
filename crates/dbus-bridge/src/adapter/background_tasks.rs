//! D-Bus adapter for `/org/desktopAssistant/BackgroundTasks` (issue
//! #116).
//!
//! Translates the bridge's `org.desktopAssistant.BackgroundTasks`
//! interface onto the daemon's `Command::*BackgroundTask*` /
//! `Command::SpawnStandaloneAgent` family. Signals fired by the
//! daemon's `Event::Task*` stream are emitted on this object path by
//! [`super::event_forwarder`]; subscribing to that stream is the
//! main binary's responsibility (it sends
//! `Command::SubscribeBackgroundTasks` over the transport at
//! startup).
//!
//! Variant encoding:
//! - `TaskView` and `TaskLogEntry` are surfaced as `a{sv}`
//!   dictionaries keyed by their serde JSON field names. Optional
//!   fields that round-trip with `skip_serializing_if =
//!   Option::is_none` simply do not appear in the dict (mirrors the
//!   JSON shape so adapters on the client side can use the same
//!   parse path for both transports).
//! - Nested values are recursive: `TaskKind` becomes a single-entry
//!   sub-dict (`{"standalone": {"name": ..., ...}}`) because serde
//!   serializes externally-tagged enums that way.
//!
//! Error mapping:
//! - The daemon returns wire-error string `"not found"` for both
//!   "id doesn't exist" and "id belongs to a different user" (the
//!   #105 hide-existence rule). We map that onto
//!   `fdo::Error::UnknownObject` so D-Bus clients can match by name.
//! - `"task is already terminal"` becomes a `Failed` with the same
//!   message — there is no closer fdo error.
//! - Everything else surfaces as `Failed` with the daemon's message
//!   verbatim.

use std::collections::HashMap;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use serde_json::Value as JsonValue;
use tracing::warn;
use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedValue, Value};
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

/// Wire-error message the daemon returns for any `NotFound` outcome
/// (issue #105 hides the difference between "unknown id" and "wrong
/// user" — both surface as this exact string).
///
/// The companion `"task is already terminal"` message (from
/// `ApiError::AlreadyTerminal`) is matched on by the `_ => Failed(msg)`
/// arm rather than a constant: there is no closer fdo error so the
/// daemon's wording is preserved verbatim.
const DAEMON_ERR_NOT_FOUND: &str = "not found";

fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => map_daemon_msg(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

/// Translate a daemon wire-error string into the closest fdo error.
/// Keeps the message verbatim so operators reading busctl output see
/// the same text logs would show.
fn map_daemon_msg(msg: String) -> fdo::Error {
    if msg == DAEMON_ERR_NOT_FOUND {
        fdo::Error::UnknownObject(msg)
    } else {
        // `AlreadyTerminal` and any other daemon error use `Failed`.
        // We could pick `AccessDenied` for the terminal case, but
        // "Failed" with the daemon's wording keeps the surface
        // honest about what happened.
        fdo::Error::Failed(msg)
    }
}

/// D-Bus adapter for background-task management. Methods translate
/// into `api::Command` dispatches; signals are emitted from the
/// transport's event broadcast by [`super::event_forwarder`].
pub struct DbusBackgroundTasksAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusBackgroundTasksAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }
}

#[interface(name = "org.desktopAssistant.BackgroundTasks")]
impl<T: BridgeTransport + 'static> DbusBackgroundTasksAdapter<T> {
    /// List the calling user's background tasks.
    ///
    /// `limit == 0` is treated as "no cap" — the wire shape carries
    /// `Option<u32>` and the registry's contract is "omit limit to
    /// return everything". Zero would otherwise be ambiguous, and
    /// `u32` has no in-band None so 0 is the natural sentinel.
    pub async fn list_tasks(
        &self,
        include_finished: bool,
        limit: u32,
    ) -> fdo::Result<Vec<HashMap<String, OwnedValue>>> {
        let result = self
            .dispatch(api::Command::ListBackgroundTasks {
                include_finished,
                limit: if limit == 0 { None } else { Some(limit) },
            })
            .await?;
        match result {
            api::CommandResult::BackgroundTasks(tasks) => {
                Ok(tasks.iter().map(task_view_to_dict).collect())
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListBackgroundTasks result: {other:?}"
            ))),
        }
    }

    /// Fetch a single task by id.
    pub async fn get_task(&self, id: &str) -> fdo::Result<HashMap<String, OwnedValue>> {
        let result = self
            .dispatch(api::Command::GetBackgroundTask { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::BackgroundTask(view) => Ok(task_view_to_dict(&view)),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetBackgroundTask result: {other:?}"
            ))),
        }
    }

    /// Request cancellation. The daemon acks synchronously; the
    /// terminal `TaskCompleted` event arrives later on the signal
    /// channel.
    pub async fn cancel_task(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::CancelBackgroundTask { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected CancelBackgroundTask result: {other:?}"
            ))),
        }
    }

    /// Page through a task's log buffer. `after_seq == 0` is the
    /// "from the beginning" sentinel (matches the wire shape's
    /// `Option<u64>` default); `limit == 0` keeps the daemon's
    /// default cap.
    pub async fn get_task_logs(
        &self,
        id: &str,
        after_seq: u64,
        limit: u32,
    ) -> fdo::Result<(Vec<HashMap<String, OwnedValue>>, u64)> {
        let result = self
            .dispatch(api::Command::GetBackgroundTaskLogs {
                id: id.to_string(),
                after_seq: if after_seq == 0 {
                    None
                } else {
                    Some(after_seq)
                },
                limit: if limit == 0 { None } else { Some(limit) },
            })
            .await?;
        match result {
            api::CommandResult::BackgroundTaskLogs { entries, next_seq } => {
                Ok((entries.iter().map(log_entry_to_dict).collect(), next_seq))
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetBackgroundTaskLogs result: {other:?}"
            ))),
        }
    }

    /// Spawn a standalone background agent.
    ///
    /// `options` is the loose `a{sv}` knob bag — today the bridge
    /// understands a single key, `tools` (`as`). Unknown keys are
    /// silently ignored so the wire shape can grow without rev'ing
    /// existing clients. `override_selection` is NOT parsed here —
    /// the JSON shape is connection-id + model-id + effort, which
    /// is awkward to fit through `a{sv}` with the right typing; the
    /// follow-up that wires KCM through this method will add it.
    pub async fn spawn_standalone(
        &self,
        name: &str,
        prompt: &str,
        options: HashMap<String, OwnedValue>,
    ) -> fdo::Result<String> {
        let tools = options.get("tools").and_then(owned_to_strings);
        let result = self
            .dispatch(api::Command::SpawnStandaloneAgent {
                name: name.to_string(),
                initial_prompt: prompt.to_string(),
                override_selection: None,
                tools,
            })
            .await?;
        match result {
            api::CommandResult::BackgroundTaskSpawned { id } => Ok(id),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SpawnStandaloneAgent result: {other:?}"
            ))),
        }
    }

    /// Emitted when a registered task transitions to `Pending`/`Running`.
    #[zbus(signal)]
    async fn task_started(
        emitter: &SignalEmitter<'_>,
        id: &str,
        task: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    /// Lightweight progress hint emitted between log entries.
    #[zbus(signal)]
    async fn task_progress(emitter: &SignalEmitter<'_>, id: &str, hint: &str) -> zbus::Result<()>;

    /// New log entry appended to a task's bounded log buffer.
    #[zbus(signal)]
    async fn task_log_appended(
        emitter: &SignalEmitter<'_>,
        id: &str,
        entry: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    /// Terminal lifecycle event for a task. `status` is the snake_case
    /// `TaskStatus` (`completed`/`failed`/`cancelled`); `last_error`
    /// is the message string, or `""` when none.
    #[zbus(signal)]
    async fn task_completed(
        emitter: &SignalEmitter<'_>,
        id: &str,
        status: &str,
        last_error: &str,
    ) -> zbus::Result<()>;
}

// ---------------------------------------------------------------------------
// Variant encoding
// ---------------------------------------------------------------------------

/// Encode a [`api::TaskView`] as a D-Bus variant dictionary keyed by
/// the JSON field names. Routes through `serde_json::Value` so the
/// keys, casing, and `skip_serializing_if` semantics match the JSON
/// surface byte-for-byte — D-Bus clients can re-use the same parse
/// code they already have for WebSocket events.
pub fn task_view_to_dict(view: &api::TaskView) -> HashMap<String, OwnedValue> {
    json_object_to_dict(serde_json::to_value(view).unwrap_or(JsonValue::Null))
}

/// Encode a [`api::TaskLogEntry`] the same way.
pub fn log_entry_to_dict(entry: &api::TaskLogEntry) -> HashMap<String, OwnedValue> {
    json_object_to_dict(serde_json::to_value(entry).unwrap_or(JsonValue::Null))
}

/// Convert a top-level `serde_json::Object` into an `a{sv}` dict.
/// Non-object input becomes an empty dict (so callers don't need to
/// branch on encoding failures — they get an obviously-empty value
/// they can surface in logs).
fn json_object_to_dict(value: JsonValue) -> HashMap<String, OwnedValue> {
    let JsonValue::Object(map) = value else {
        warn!("background_tasks: value to encode is not a JSON object");
        return HashMap::new();
    };
    let mut out = HashMap::with_capacity(map.len());
    for (key, val) in map {
        let Some(ov) = json_value_to_owned(val) else {
            // Skip nulls so the dict mirrors `skip_serializing_if =
            // Option::is_none` JSON output.
            continue;
        };
        out.insert(key, ov);
    }
    out
}

/// Recursively encode a `serde_json::Value` as a D-Bus `OwnedValue`.
/// `Null` returns `None` so the caller can drop the key entirely.
fn json_value_to_owned(value: JsonValue) -> Option<OwnedValue> {
    match value {
        JsonValue::Null => None,
        JsonValue::Bool(b) => OwnedValue::try_from(Value::from(b)).ok(),
        JsonValue::Number(n) => {
            // JSON numbers carry no signedness; serde_json itself
            // picks between i64 and u64 based on whether the value
            // is negative. We match that classification so the
            // D-Bus type lines up with the JSON-decoded shape on
            // the other side: positive integers come through as
            // `u64` (matches `TaskLogEntry::seq` and friends),
            // negatives come through as `i64`, fractions as `f64`.
            if let Some(u) = n.as_u64() {
                OwnedValue::try_from(Value::from(u)).ok()
            } else if let Some(i) = n.as_i64() {
                OwnedValue::try_from(Value::from(i)).ok()
            } else if let Some(f) = n.as_f64() {
                OwnedValue::try_from(Value::from(f)).ok()
            } else {
                None
            }
        }
        JsonValue::String(s) => OwnedValue::try_from(Value::from(s)).ok(),
        JsonValue::Array(items) => {
            // D-Bus arrays are homogeneous. We pack JSON arrays as
            // `av` (array of variant) so heterogeneous JSON arrays
            // round-trip cleanly; the wrapper variant tags each
            // element with its own signature.
            let mut packed: Vec<Value<'static>> = Vec::with_capacity(items.len());
            for item in items {
                if let Some(ov) = json_value_to_owned(item) {
                    // `Value::from(OwnedValue)` wraps as a variant.
                    packed.push(Value::from(ov));
                }
            }
            OwnedValue::try_from(Value::from(packed)).ok()
        }
        JsonValue::Object(map) => {
            // Nested objects become `a{sv}` dicts (variant of dict).
            let mut nested: HashMap<String, OwnedValue> = HashMap::with_capacity(map.len());
            for (k, v) in map {
                if let Some(ov) = json_value_to_owned(v) {
                    nested.insert(k, ov);
                }
            }
            // Wrap the HashMap as a Value::Dict so it can be carried
            // in an `a{sv}` slot. The `From<HashMap>` impl on Value
            // does exactly this.
            OwnedValue::try_from(Value::from(nested)).ok()
        }
    }
}

/// Best-effort coercion of an `OwnedValue` carrying `as` (array of
/// strings) into `Vec<String>`. Returns `None` if the value is not
/// the expected shape — the caller falls back to "no tools".
fn owned_to_strings(value: &OwnedValue) -> Option<Vec<String>> {
    let v: Value<'_> = (**value).try_clone().ok()?;
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr.iter() {
                match item {
                    Value::Str(s) => out.push(s.as_str().to_string()),
                    Value::Value(boxed) => {
                        if let Value::Str(s) = boxed.as_ref() {
                            out.push(s.as_str().to_string());
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_value_to_owned_null_returns_none() {
        assert!(json_value_to_owned(JsonValue::Null).is_none());
    }

    #[test]
    fn json_value_to_owned_bool_round_trips() {
        let ov = json_value_to_owned(JsonValue::Bool(true)).unwrap();
        let back: bool = ov.try_into().unwrap();
        assert!(back);
    }

    #[test]
    fn json_value_to_owned_i64_round_trips() {
        let ov = json_value_to_owned(JsonValue::Number(serde_json::Number::from(-42_i64))).unwrap();
        let back: i64 = ov.try_into().unwrap();
        assert_eq!(back, -42);
    }

    #[test]
    fn json_value_to_owned_u64_round_trips() {
        let ov = json_value_to_owned(JsonValue::Number(serde_json::Number::from(42_u64))).unwrap();
        let back: u64 = ov.try_into().unwrap();
        assert_eq!(back, 42);
    }

    #[test]
    fn json_value_to_owned_string_round_trips() {
        let ov = json_value_to_owned(JsonValue::String("hi".into())).unwrap();
        let back: String = ov.try_into().unwrap();
        assert_eq!(back, "hi");
    }

    #[test]
    fn json_value_to_owned_nested_object_becomes_dict() {
        let val = serde_json::json!({"a": {"b": 1}});
        let ov = json_value_to_owned(val).unwrap();
        let outer: HashMap<String, OwnedValue> = ov.try_into().unwrap();
        let inner: HashMap<String, OwnedValue> =
            outer.get("a").cloned().unwrap().try_into().unwrap();
        // `1` is non-negative so the encoder picks `u64` (matches
        // serde_json's classification — see `json_value_to_owned`).
        let b: u64 = inner.get("b").cloned().unwrap().try_into().unwrap();
        assert_eq!(b, 1);
    }

    #[test]
    fn map_daemon_msg_routes_not_found_to_unknown_object() {
        let err = map_daemon_msg("not found".to_string());
        assert!(matches!(err, fdo::Error::UnknownObject(_)));
    }

    #[test]
    fn map_daemon_msg_routes_terminal_to_failed_with_message() {
        let err = map_daemon_msg("task is already terminal".to_string());
        match err {
            fdo::Error::Failed(msg) => assert_eq!(msg, "task is already terminal"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn owned_to_strings_handles_array_of_strings() {
        // Shape clients actually send: an `as` array whose elements
        // are bare `Value::Str` (not variant-wrapped).
        let arr: Vec<Value<'_>> = vec![Value::from("a"), Value::from("b")];
        let ov: OwnedValue = Value::from(arr).try_into().unwrap();
        let parsed = owned_to_strings(&ov).expect("parses");
        assert_eq!(parsed, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn owned_to_strings_rejects_non_array() {
        let ov: OwnedValue = Value::from("not an array").try_into().unwrap();
        assert!(owned_to_strings(&ov).is_none());
    }
}
