//! D-Bus adapter for `/org/desktopAssistant/BackgroundTasks` (issue #116).
//!
//! Skeleton — methods are wired but intentionally return `Failed("not
//! implemented yet")` errors so the TDD step lands tests in a known
//! red state without breaking the build. The implementation commit
//! replaces the stubs with real translations against the
//! [`BridgeTransport`].

use std::collections::HashMap;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::object_server::SignalEmitter;
use zbus::zvariant::OwnedValue;
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

#[allow(dead_code)]
fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => fdo::Error::Failed(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

/// D-Bus adapter for background-task management. Methods translate
/// into `api::Command` dispatches; signals are emitted from the
/// transport's event broadcast by [`super::event_forwarder`].
pub struct DbusBackgroundTasksAdapter<T: BridgeTransport + 'static> {
    #[allow(dead_code)]
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusBackgroundTasksAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }
}

#[interface(name = "org.desktopAssistant.BackgroundTasks")]
impl<T: BridgeTransport + 'static> DbusBackgroundTasksAdapter<T> {
    pub async fn list_tasks(
        &self,
        _include_finished: bool,
        _limit: u32,
    ) -> fdo::Result<Vec<HashMap<String, OwnedValue>>> {
        Err(fdo::Error::Failed("not implemented yet".to_string()))
    }

    pub async fn get_task(&self, _id: &str) -> fdo::Result<HashMap<String, OwnedValue>> {
        Err(fdo::Error::Failed("not implemented yet".to_string()))
    }

    pub async fn cancel_task(&self, _id: &str) -> fdo::Result<()> {
        Err(fdo::Error::Failed("not implemented yet".to_string()))
    }

    pub async fn get_task_logs(
        &self,
        _id: &str,
        _after_seq: u64,
        _limit: u32,
    ) -> fdo::Result<(Vec<HashMap<String, OwnedValue>>, u64)> {
        Err(fdo::Error::Failed("not implemented yet".to_string()))
    }

    pub async fn spawn_standalone(
        &self,
        _name: &str,
        _prompt: &str,
        _options: HashMap<String, OwnedValue>,
    ) -> fdo::Result<String> {
        Err(fdo::Error::Failed("not implemented yet".to_string()))
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
    async fn task_progress(
        emitter: &SignalEmitter<'_>,
        id: &str,
        hint: &str,
    ) -> zbus::Result<()>;

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

/// Encode a [`api::TaskView`] as a D-Bus variant dictionary keyed by
/// the JSON field names. Stub — returns an empty dict until the
/// implementation lands.
#[allow(dead_code)]
pub fn task_view_to_dict(_view: &api::TaskView) -> HashMap<String, OwnedValue> {
    HashMap::new()
}

/// Encode a [`api::TaskLogEntry`] as a D-Bus variant dictionary keyed
/// by JSON field names. Stub.
#[allow(dead_code)]
pub fn log_entry_to_dict(_entry: &api::TaskLogEntry) -> HashMap<String, OwnedValue> {
    HashMap::new()
}
