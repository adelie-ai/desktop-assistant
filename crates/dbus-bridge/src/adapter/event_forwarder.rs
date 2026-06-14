//! Pump the daemon's signal stream into D-Bus signals.
//!
//! Subscribes to the client-common [`Connector`]'s [`SignalEvent`] stream (#316,
//! the same stream every UDS/WS client consumes) and dispatches each event to
//! the matching object path on a [`zbus::Connection`]. Translation mirrors the
//! in-process daemon's signal vocabulary one-for-one:
//!
//! | Signal event       | D-Bus signal                              |
//! | ------------------ | ----------------------------------------- |
//! | `Chunk`            | `Conversations.ResponseChunk`             |
//! | `Complete`         | `Conversations.ResponseComplete`          |
//! | `Error`            | `Conversations.ResponseError`             |
//! | `TaskStarted`      | `BackgroundTasks.TaskStarted` (#116)      |
//! | `TaskProgress`     | `BackgroundTasks.TaskProgress` (#116)     |
//! | `TaskLogAppended`  | `BackgroundTasks.TaskLogAppended` (#116)  |
//! | `TaskCompleted`    | `BackgroundTasks.TaskCompleted` (#116)    |
//! | other              | dropped (see #367 for the parity follow-up)|
//!
//! `Settings.ConfigChanged` is **not** forwarded here: the decoded
//! [`SignalEvent`] stream carries no config event, and the bridge's
//! `Settings.set_config` adapter already emits `ConfigChanged` directly after a
//! successful write — so there is no regression (the in-process surface only
//! ever delivered a config change to the connection that made it, which over the
//! bridge is the bridge's own `set_config`).
//!
//! ## Reconnect (#316)
//!
//! On a daemon restart the Connector drops the underlying socket, delivers a
//! terminal [`SignalEvent::Disconnected`] to this subscriber (closing it), and
//! reconnects in the background. This loop re-`subscribe()`s for a fresh stream
//! and re-issues `SubscribeBackgroundTasks` (the Connector replays only
//! client-tool registrations, not this subscription), so `Task*` signals resume
//! once the daemon is back. The conversation response signals resume on their
//! own — they ride whatever turn a D-Bus client drives next.
//!
//! Signals are emitted via `zbus::Connection::emit_signal` rather than the
//! adapter types' generated helpers (made private by `#[interface]`); the
//! forwarder emits from a context that doesn't own a typed adapter reference.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_client_common::{Connector, SignalEvent};
use tracing::{debug, warn};
use zbus::Connection;
use zbus::zvariant::OwnedValue;

use super::background_tasks::{log_entry_to_dict, task_view_to_dict};
use super::paths;

const CONV_INTERFACE: &str = "org.desktopAssistant.Conversations";
const BG_INTERFACE: &str = "org.desktopAssistant.BackgroundTasks";

/// The result of translating one [`SignalEvent`] — the testable surface; `run`
/// is the wiring loop.
#[derive(Debug, Clone, PartialEq)]
pub enum ForwardAction {
    ResponseChunk {
        conversation_id: String,
        request_id: String,
        chunk: String,
    },
    ResponseComplete {
        conversation_id: String,
        request_id: String,
        full_response: String,
    },
    ResponseError {
        conversation_id: String,
        request_id: String,
        error: String,
    },
    /// Task is now `Pending`/`Running`. `task` is the JSON-keyed `TaskView`
    /// encoded as `a{sv}`.
    TaskStarted {
        id: String,
        task: HashMap<String, OwnedValue>,
    },
    /// Lightweight progress hint. `hint` is `""` when the event carried `None`.
    TaskProgress { id: String, hint: String },
    /// A new log entry, encoded as `a{sv}`.
    TaskLogAppended {
        id: String,
        entry: HashMap<String, OwnedValue>,
    },
    /// Terminal: `status` is the snake_case `TaskStatus`; `last_error` is `""`
    /// when none.
    TaskCompleted {
        id: String,
        status: String,
        last_error: String,
    },
    /// No matching D-Bus signal in this bridge (v1). Recorded so tests can
    /// assert a deliberate ignore rather than a missed translation. Forwarding
    /// these to D-Bus (UserMessageAdded / ConversationListChanged / …) for full
    /// UDS/WS parity is the #367 follow-up.
    Ignored { kind: &'static str },
}

/// Pure translator: one signal event → a D-Bus action. Sync / no-side-effects so
/// tests can assert each variant in isolation.
pub fn translate(event: SignalEvent) -> ForwardAction {
    match event {
        SignalEvent::Chunk {
            conversation_id,
            request_id,
            chunk,
        } => ForwardAction::ResponseChunk {
            conversation_id,
            request_id,
            chunk,
        },
        SignalEvent::Complete {
            conversation_id,
            request_id,
            full_response,
        } => ForwardAction::ResponseComplete {
            conversation_id,
            request_id,
            full_response,
        },
        SignalEvent::Error {
            conversation_id,
            request_id,
            error,
        } => ForwardAction::ResponseError {
            conversation_id,
            request_id,
            error,
        },
        SignalEvent::TaskStarted { task } => {
            let id = task.id.0.clone();
            ForwardAction::TaskStarted {
                id,
                task: task_view_to_dict(&task),
            }
        }
        SignalEvent::TaskProgress { id, progress_hint } => ForwardAction::TaskProgress {
            id,
            hint: progress_hint.unwrap_or_default(),
        },
        SignalEvent::TaskLogAppended { id, entry } => ForwardAction::TaskLogAppended {
            id,
            entry: log_entry_to_dict(&entry),
        },
        SignalEvent::TaskCompleted {
            id,
            status,
            last_error,
        } => ForwardAction::TaskCompleted {
            id,
            status: task_status_str(status).to_string(),
            last_error: last_error.unwrap_or_default(),
        },
        // --- deliberately not forwarded in v1 (see the module docs + #367) ---
        SignalEvent::UserMessageAdded { .. } => ForwardAction::Ignored {
            kind: "user_message_added",
        },
        SignalEvent::Status { .. } => ForwardAction::Ignored {
            kind: "assistant_status",
        },
        SignalEvent::ContextUsage { .. } => ForwardAction::Ignored {
            kind: "context_usage",
        },
        SignalEvent::TitleChanged { .. } => ForwardAction::Ignored {
            kind: "conversation_title_changed",
        },
        SignalEvent::ConversationListChanged { .. } => ForwardAction::Ignored {
            kind: "conversation_list_changed",
        },
        SignalEvent::ConversationWarning { .. } => ForwardAction::Ignored {
            kind: "conversation_warning",
        },
        SignalEvent::ScratchpadChanged { .. } => ForwardAction::Ignored {
            kind: "scratchpad_changed",
        },
        SignalEvent::ClientToolCall { .. } => ForwardAction::Ignored {
            kind: "client_tool_call",
        },
        // Control signal handled by `run` before it reaches `translate`; mapped
        // here only for match exhaustiveness.
        SignalEvent::Disconnected { .. } => ForwardAction::Ignored {
            kind: "disconnected",
        },
    }
}

/// Run the forwarder until `shutdown` resolves. Survives daemon restarts: on a
/// [`SignalEvent::Disconnected`] it re-subscribes for a fresh stream and
/// re-issues the background-task subscription once the Connector reconnects.
pub async fn run<F: std::future::Future<Output = ()> + Send + 'static>(
    connector: Arc<Connector>,
    connection: Connection,
    shutdown: F,
) {
    tokio::pin!(shutdown);
    let mut events = connector.subscribe();
    // Initial background-task subscription (retries until the daemon answers).
    spawn_background_task_subscription(&connector);

    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                debug!("event forwarder shutting down");
                return;
            }
            recv = events.recv() => match recv {
                Some(SignalEvent::Disconnected { reason }) => {
                    debug!("event forwarder: transport dropped ({reason}); re-subscribing");
                    events = connector.subscribe();
                    spawn_background_task_subscription(&connector);
                }
                Some(event) => emit(&connection, translate(event)).await,
                None => {
                    // Sender dropped without a Disconnected (shouldn't happen while
                    // the bridge holds the Connector). Re-subscribe defensively.
                    events = connector.subscribe();
                }
            }
        }
    }
}

/// Spawn a detached task that issues `SubscribeBackgroundTasks`, retrying with
/// backoff until the (re)connection answers. Holds only a `Weak<Connector>` so a
/// pending retry can't keep the Connector — and thus the bridge's reconnect
/// supervisor — alive past shutdown.
fn spawn_background_task_subscription(connector: &Arc<Connector>) {
    let weak = Arc::downgrade(connector);
    tokio::spawn(async move {
        let mut backoff = Duration::from_millis(100);
        loop {
            let Some(connector) = weak.upgrade() else {
                return; // Connector gone — nothing to subscribe.
            };
            let outcome = match connector.client().as_commands() {
                Some(commands) => commands
                    .send_command(api::Command::SubscribeBackgroundTasks)
                    .await
                    .map(|_| ()),
                None => return, // no command channel (not a socket transport)
            };
            drop(connector); // don't hold the Arc across the sleep
            match outcome {
                Ok(()) => {
                    debug!("event forwarder: subscribed to background-task events");
                    return;
                }
                Err(e) => {
                    debug!("event forwarder: background-task subscribe failed ({e}); retrying");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(10));
                }
            }
        }
    });
}

/// Translate and emit one signal event as its D-Bus signal. The single-event
/// seam `run` uses internally, exposed so integration tests can drive the emit
/// path over a p2p connection without standing up a full Connector.
pub async fn forward_one(connection: &Connection, event: SignalEvent) {
    emit(connection, translate(event)).await;
}

async fn emit(connection: &Connection, action: ForwardAction) {
    match action {
        ForwardAction::ResponseChunk {
            conversation_id,
            request_id,
            chunk,
        } => {
            let body = (conversation_id, request_id, chunk);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::CONVERSATIONS,
                    CONV_INTERFACE,
                    "ResponseChunk",
                    &body,
                )
                .await
            {
                warn!("response_chunk emit failed: {e}");
            }
        }
        ForwardAction::ResponseComplete {
            conversation_id,
            request_id,
            full_response,
        } => {
            let body = (conversation_id, request_id, full_response);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::CONVERSATIONS,
                    CONV_INTERFACE,
                    "ResponseComplete",
                    &body,
                )
                .await
            {
                warn!("response_complete emit failed: {e}");
            }
        }
        ForwardAction::ResponseError {
            conversation_id,
            request_id,
            error,
        } => {
            let body = (conversation_id, request_id, error);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::CONVERSATIONS,
                    CONV_INTERFACE,
                    "ResponseError",
                    &body,
                )
                .await
            {
                warn!("response_error emit failed: {e}");
            }
        }
        ForwardAction::TaskStarted { id, task } => {
            let body = (id, task);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::BACKGROUND_TASKS,
                    BG_INTERFACE,
                    "TaskStarted",
                    &body,
                )
                .await
            {
                warn!("task_started emit failed: {e}");
            }
        }
        ForwardAction::TaskProgress { id, hint } => {
            let body = (id, hint);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::BACKGROUND_TASKS,
                    BG_INTERFACE,
                    "TaskProgress",
                    &body,
                )
                .await
            {
                warn!("task_progress emit failed: {e}");
            }
        }
        ForwardAction::TaskLogAppended { id, entry } => {
            let body = (id, entry);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::BACKGROUND_TASKS,
                    BG_INTERFACE,
                    "TaskLogAppended",
                    &body,
                )
                .await
            {
                warn!("task_log_appended emit failed: {e}");
            }
        }
        ForwardAction::TaskCompleted {
            id,
            status,
            last_error,
        } => {
            let body = (id, status, last_error);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::BACKGROUND_TASKS,
                    BG_INTERFACE,
                    "TaskCompleted",
                    &body,
                )
                .await
            {
                warn!("task_completed emit failed: {e}");
            }
        }
        ForwardAction::Ignored { kind } => {
            debug!("event forwarder ignoring kind={kind}");
        }
    }
}

/// Snake-case wire string for `api::TaskStatus`, matching the enum's
/// `#[serde(rename_all = "snake_case")]` so D-Bus clients see the same
/// vocabulary as the JSON/WS surface.
fn task_status_str(status: api::TaskStatus) -> &'static str {
    match status {
        api::TaskStatus::Pending => "pending",
        api::TaskStatus::Running => "running",
        api::TaskStatus::Completed => "completed",
        api::TaskStatus::Failed => "failed",
        api::TaskStatus::Cancelled => "cancelled",
    }
}
