//! Pump `WsFrame::Event`s from the transport into D-Bus signals.
//!
//! Subscribes to [`BridgeTransport::subscribe_events`] and dispatches
//! each event to the matching object path on a [`zbus::Connection`].
//! Translation mirrors the in-process daemon's signal vocabulary
//! one-for-one:
//!
//! | Wire event              | D-Bus signal                                  |
//! | ----------------------- | --------------------------------------------- |
//! | `AssistantDelta`        | `Conversations.ResponseChunk`                 |
//! | `AssistantCompleted`    | `Conversations.ResponseComplete`              |
//! | `AssistantError`        | `Conversations.ResponseError`                 |
//! | `ConfigChanged`         | `Settings.ConfigChanged`                      |
//! | `TaskStarted`           | `BackgroundTasks.TaskStarted` (#116)          |
//! | `TaskProgress`          | `BackgroundTasks.TaskProgress` (#116)         |
//! | `TaskLogAppended`       | `BackgroundTasks.TaskLogAppended` (#116)      |
//! | `TaskCompleted`         | `BackgroundTasks.TaskCompleted` (#116)        |
//! | other (`AssistantStatus`, ...) | dropped                                |
//!
//! Returned future runs until the inbound channel closes or
//! `shutdown` resolves.
//!
//! Signals are emitted via `zbus::Connection::emit_signal` rather than
//! the auto-generated signal helpers on the adapter types — those
//! helpers are made private by the `#[interface]` macro, and the
//! forwarder needs to emit from a context that doesn't own a typed
//! `&Adapter` reference. Using `emit_signal` directly also keeps the
//! cross-module surface narrow.

use desktop_assistant_api_model as api;
use tokio::sync::broadcast;
use tracing::{debug, warn};
use zbus::Connection;

use std::collections::HashMap;

use zbus::zvariant::OwnedValue;

use super::background_tasks::{log_entry_to_dict, task_view_to_dict};
use super::paths;
use super::settings::{ConfigData, config_data_from_event};

const CONV_INTERFACE: &str = "org.desktopAssistant.Conversations";
const SETTINGS_INTERFACE: &str = "org.desktopAssistant.Settings";
const BG_INTERFACE: &str = "org.desktopAssistant.BackgroundTasks";

/// Map a single wire event onto its D-Bus signal. Public so tests can
/// drive it without a live zbus connection — the `ForwardAction` enum
/// is the testable surface; `run` is the wiring loop.
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
    ConfigChanged {
        config: ConfigData,
    },
    /// Task is now `Pending`/`Running`. `task` is the JSON-keyed
    /// `TaskView` encoded as `a{sv}`.
    TaskStarted {
        id: String,
        task: HashMap<String, OwnedValue>,
    },
    /// Lightweight progress hint between log entries. `hint` is `""`
    /// when the wire event carried `None`.
    TaskProgress {
        id: String,
        hint: String,
    },
    /// A new log entry was appended to the task's bounded buffer.
    /// `entry` is the JSON-keyed `TaskLogEntry` encoded as `a{sv}`.
    TaskLogAppended {
        id: String,
        entry: HashMap<String, OwnedValue>,
    },
    /// Terminal event: `status` is the snake_case `TaskStatus`;
    /// `last_error` is `""` when none.
    TaskCompleted {
        id: String,
        status: String,
        last_error: String,
    },
    /// Event has no matching D-Bus signal in this bridge. Recorded so
    /// tests can assert "we deliberately ignored X" without that being
    /// confused for a translation bug.
    Ignored {
        kind: &'static str,
    },
}

/// Pure translator: wire event → D-Bus action. Pure / sync /
/// no-side-effects so tests can assert each variant in isolation.
pub fn translate(event: api::Event) -> ForwardAction {
    match event {
        api::Event::AssistantDelta {
            conversation_id,
            request_id,
            chunk,
        } => ForwardAction::ResponseChunk {
            conversation_id,
            request_id,
            chunk,
        },
        api::Event::AssistantCompleted {
            conversation_id,
            request_id,
            full_response,
        } => ForwardAction::ResponseComplete {
            conversation_id,
            request_id,
            full_response,
        },
        api::Event::AssistantError {
            conversation_id,
            request_id,
            error,
        } => ForwardAction::ResponseError {
            conversation_id,
            request_id,
            error,
        },
        api::Event::ConfigChanged { config } => ForwardAction::ConfigChanged {
            config: config_data_from_event(&config),
        },
        api::Event::AssistantStatus { .. } => ForwardAction::Ignored {
            kind: "assistant_status",
        },
        api::Event::ConversationTitleChanged { .. } => ForwardAction::Ignored {
            kind: "conversation_title_changed",
        },
        api::Event::ConversationWarningEmitted { .. } => ForwardAction::Ignored {
            kind: "conversation_warning_emitted",
        },
        api::Event::TaskStarted { task } => {
            let id = task.id.0.clone();
            let dict = task_view_to_dict(&task);
            ForwardAction::TaskStarted { id, task: dict }
        }
        api::Event::TaskProgress { id, progress_hint } => ForwardAction::TaskProgress {
            id,
            hint: progress_hint.unwrap_or_default(),
        },
        api::Event::TaskLogAppended { id, entry } => {
            let dict = log_entry_to_dict(&entry);
            ForwardAction::TaskLogAppended { id, entry: dict }
        }
        api::Event::TaskCompleted {
            id,
            status,
            last_error,
        } => ForwardAction::TaskCompleted {
            id,
            status: task_status_str(status).to_string(),
            last_error: last_error.unwrap_or_default(),
        },
        // Conversation scratchpad (issue #190): not forwarded over D-Bus —
        // the scratchpad side pane is a WebSocket/UDS-protocol concern
        // (adele-gtk subscribes via the command channel). A D-Bus client that
        // wants live scratchpad updates would need its own forwarding arm.
        api::Event::ScratchpadChanged { .. } => ForwardAction::Ignored {
            kind: "scratchpad_changed",
        },
        // Client-side tool execution (issue #107): the bridge does not
        // forward `ClientToolCall` to D-Bus subscribers because client-
        // side execution is a WebSocket-protocol concern. The D-Bus
        // bridge IS a client — if the daemon ever picks a tool that's
        // registered on a D-Bus-attached client, the bridge would need
        // its own dispatch path; today the bridge does not register
        // any client-local tools, so this is unreachable in practice.
        api::Event::ClientToolCall { .. } => ForwardAction::Ignored {
            kind: "client_tool_call",
        },
    }
}

/// Run the forwarder loop until the inbound channel closes or
/// `shutdown` resolves. Lagged subscribers drop a warning and continue
/// (broadcast semantics) — losing an old D-Bus signal is better than
/// blocking the demux task.
pub async fn run<F: std::future::Future<Output = ()> + Send + 'static>(
    mut events: broadcast::Receiver<api::Event>,
    connection: Connection,
    shutdown: F,
) {
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                debug!("event forwarder shutting down");
                return;
            }
            recv = events.recv() => {
                match recv {
                    Ok(event) => emit(&connection, translate(event)).await,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("event forwarder lagged; dropped {n} events");
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("event channel closed; forwarder exiting");
                        return;
                    }
                }
            }
        }
    }
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
        ForwardAction::ConfigChanged { config } => {
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    None,
                    paths::SETTINGS,
                    SETTINGS_INTERFACE,
                    "ConfigChanged",
                    &config,
                )
                .await
            {
                warn!("config_changed emit failed: {e}");
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

/// Snake-case wire string for `api::TaskStatus`. Mirrors the
/// `#[serde(rename_all = "snake_case")]` attribute on the enum so
/// D-Bus clients see the same wire vocabulary as the JSON/WS surface.
fn task_status_str(status: api::TaskStatus) -> &'static str {
    match status {
        api::TaskStatus::Pending => "pending",
        api::TaskStatus::Running => "running",
        api::TaskStatus::Completed => "completed",
        api::TaskStatus::Failed => "failed",
        api::TaskStatus::Cancelled => "cancelled",
    }
}
