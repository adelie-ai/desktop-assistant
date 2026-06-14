//! Pump the daemon's signal stream into D-Bus signals.
//!
//! Subscribes to the client-common [`Connector`]'s [`SignalEvent`] stream (#316,
//! the same stream every UDS/WS client consumes) and dispatches each event to
//! the matching object path on a [`zbus::Connection`]. Translation mirrors the
//! in-process daemon's signal vocabulary one-for-one:
//!
//! | Signal event            | D-Bus signal                                   |
//! | ----------------------- | ---------------------------------------------- |
//! | `Chunk`                 | `Conversations.ResponseChunk`                  |
//! | `Complete`              | `Conversations.ResponseComplete`               |
//! | `Error`                 | `Conversations.ResponseError`                  |
//! | `UserMessageAdded`      | `Conversations.UserMessageAdded` (#367)        |
//! | `ConversationListChanged`| `Conversations.ConversationListChanged` (#367)|
//! | `ClientToolCall`        | `Conversations.ClientToolCall` (#320)          |
//! | `TaskStarted`           | `BackgroundTasks.TaskStarted` (#116)           |
//! | `TaskProgress`          | `BackgroundTasks.TaskProgress` (#116)          |
//! | `TaskLogAppended`       | `BackgroundTasks.TaskLogAppended` (#116)       |
//! | `TaskCompleted`         | `BackgroundTasks.TaskCompleted` (#116)         |
//! | other                   | dropped (Status/ContextUsage/Title/Warning/Scratch)|
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
    /// A user message was committed and a turn started in a conversation the
    /// recipient is viewing (#367). Unicast to a per-sender session: it arrives
    /// via that session's `SubscribeConversations` fan-out — including turns the
    /// recipient did NOT initiate (a voice turn, or another client) — so a client
    /// can render the user bubble live. The initiator dedupes on `request_id`.
    UserMessageAdded {
        conversation_id: String,
        request_id: String,
        content: String,
    },
    /// The user's conversation list changed — created/renamed/deleted/(un)archived
    /// by any client or the voice daemon (#367). Broadcast on the shared per-user
    /// stream so every D-Bus client refreshes its sidebar; carries only the
    /// affected `conversation_id` (clients re-fetch the list).
    ConversationListChanged { conversation_id: String },
    /// A turn suspended on a client-side tool call (#320). Unicast to the session
    /// that registered the tool (= the session driving the turn), so the caller
    /// runs the tool and posts the outcome back via a `ClientToolResult` command
    /// carrying the same `task_id` + `tool_call_id`. `arguments_json` is the tool
    /// input serialized to JSON (the wire event carries a `serde_json::Value`).
    ClientToolCall {
        task_id: String,
        conversation_id: String,
        tool_call_id: String,
        tool_name: String,
        arguments_json: String,
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

impl ForwardAction {
    /// Whether this action belongs on the per-user **broadcast** stream (the
    /// shared connection, `destination = None`) rather than a per-sender
    /// **unicast** session stream. `Task*` and `ConversationListChanged` ride the
    /// daemon's per-user broadcast (the `SubscribeBackgroundTasks` tap the shared
    /// connection holds); the response stream and `UserMessageAdded` are
    /// per-conversation fan-out delivered to a session's own connection.
    ///
    /// Used by [`run_unicast`] to defensively skip any broadcast-class action: a
    /// per-sender session never *should* receive one (it holds no
    /// `SubscribeBackgroundTasks`), but if daemon delivery ever changed, this
    /// keeps a list-change from being re-emitted once per session on top of the
    /// shared broadcast.
    fn is_broadcast(&self) -> bool {
        matches!(
            self,
            ForwardAction::TaskStarted { .. }
                | ForwardAction::TaskProgress { .. }
                | ForwardAction::TaskLogAppended { .. }
                | ForwardAction::TaskCompleted { .. }
                | ForwardAction::ConversationListChanged { .. }
        )
    }
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
        // #367: forwarded for full UDS/WS parity. `UserMessageAdded` reaches a
        // per-sender session via its `SubscribeConversations` fan-out (unicast);
        // `ConversationListChanged` rides the shared per-user broadcast.
        SignalEvent::UserMessageAdded {
            conversation_id,
            request_id,
            content,
        } => ForwardAction::UserMessageAdded {
            conversation_id,
            request_id,
            content,
        },
        SignalEvent::ConversationListChanged { conversation_id } => {
            ForwardAction::ConversationListChanged { conversation_id }
        }
        // #320: the turn suspended on a client tool. Unicast to the registrant's
        // session; the args ride as a JSON string (the wire event is a Value).
        SignalEvent::ClientToolCall {
            task_id,
            conversation_id,
            tool_call_id,
            tool_name,
            arguments,
        } => ForwardAction::ClientToolCall {
            task_id,
            conversation_id,
            tool_call_id,
            tool_name,
            arguments_json: arguments.to_string(),
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
        // --- not (yet) forwarded: no D-Bus signal for these. Status/ContextUsage/
        // TitleChanged are richer-parity follow-ups; ClientToolCall is #320;
        // recorded as a deliberate ignore rather than a missed translation. ---
        SignalEvent::Status { .. } => ForwardAction::Ignored {
            kind: "assistant_status",
        },
        SignalEvent::ContextUsage { .. } => ForwardAction::Ignored {
            kind: "context_usage",
        },
        SignalEvent::TitleChanged { .. } => ForwardAction::Ignored {
            kind: "conversation_title_changed",
        },
        SignalEvent::ConversationWarning { .. } => ForwardAction::Ignored {
            kind: "conversation_warning",
        },
        SignalEvent::ScratchpadChanged { .. } => ForwardAction::Ignored {
            kind: "scratchpad_changed",
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
                // Broadcast (destination `None`): this shared connection carries
                // the per-user stream (`Task*`, and post-#367 the conversation-list
                // signal). Per-sender turn responses ride `run_unicast` instead.
                Some(event) => emit(&connection, None, translate(event)).await,
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
///
/// `destination` mirrors [`emit`]: `None` broadcasts, `Some(unique_name)`
/// unicasts to one D-Bus sender (the per-sender session path).
pub async fn forward_one(connection: &Connection, destination: Option<&str>, event: SignalEvent) {
    emit(connection, destination, translate(event)).await;
}

/// Run a **per-sender** forwarder until the task is aborted (on session
/// eviction). Unlike [`run`], this pumps one [`SenderSession`](crate::session)'s
/// own daemon connection — which carries only that session's turn responses
/// (`AssistantDelta`/`Completed`/`Error`), because the sub-session holds no
/// `SubscribeBackgroundTasks`/`SubscribeConversations` registration of its own —
/// and emits each as a signal **unicast to `destination`** (the sender's unique
/// bus name). So a turn driven by one D-Bus caller streams back only to that
/// caller, never broadcast across the session bus.
///
/// No shutdown future and no background-task subscription: the session owns this
/// task's `JoinHandle` and aborts it on eviction, and there is nothing to
/// re-subscribe across a reconnect (the sub-session carries no subscriptions).
pub async fn run_unicast(connector: Arc<Connector>, connection: Connection, destination: String) {
    let mut events = connector.subscribe();
    loop {
        match events.recv().await {
            Some(SignalEvent::Disconnected { reason }) => {
                debug!(
                    "unicast forwarder for {destination}: transport dropped ({reason}); re-subscribing"
                );
                events = connector.subscribe();
            }
            Some(event) => {
                let action = translate(event);
                if action.is_broadcast() {
                    // A per-sender session holds no `SubscribeBackgroundTasks`, so
                    // it should never receive a broadcast-class event; if it ever
                    // did, the shared forwarder already broadcasts it — don't also
                    // unicast a duplicate to this one sender.
                    debug!("unicast forwarder for {destination}: skipping broadcast-class action");
                } else {
                    emit(&connection, Some(&destination), action).await;
                }
            }
            None => events = connector.subscribe(),
        }
    }
}

/// Emit `action` as its D-Bus signal. `destination` is the signal's intended
/// recipient: `None` broadcasts (the shared per-user stream — `Task*` and, post
/// #367, the conversation-list signal), `Some(unique_name)` **unicasts** to one
/// D-Bus sender (a per-sender session's own turn responses — #367/#320). A
/// unicast signal still matches an ordinary member match rule at the recipient,
/// so a client subscribed the usual way receives it transparently.
async fn emit(connection: &Connection, destination: Option<&str>, action: ForwardAction) {
    match action {
        ForwardAction::ResponseChunk {
            conversation_id,
            request_id,
            chunk,
        } => {
            let body = (conversation_id, request_id, chunk);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    destination,
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
                    destination,
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
                    destination,
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
        ForwardAction::UserMessageAdded {
            conversation_id,
            request_id,
            content,
        } => {
            let body = (conversation_id, request_id, content);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    destination,
                    paths::CONVERSATIONS,
                    CONV_INTERFACE,
                    "UserMessageAdded",
                    &body,
                )
                .await
            {
                warn!("user_message_added emit failed: {e}");
            }
        }
        ForwardAction::ConversationListChanged { conversation_id } => {
            let body = (conversation_id,);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    destination,
                    paths::CONVERSATIONS,
                    CONV_INTERFACE,
                    "ConversationListChanged",
                    &body,
                )
                .await
            {
                warn!("conversation_list_changed emit failed: {e}");
            }
        }
        ForwardAction::ClientToolCall {
            task_id,
            conversation_id,
            tool_call_id,
            tool_name,
            arguments_json,
        } => {
            let body = (
                task_id,
                conversation_id,
                tool_call_id,
                tool_name,
                arguments_json,
            );
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    destination,
                    paths::CONVERSATIONS,
                    CONV_INTERFACE,
                    "ClientToolCall",
                    &body,
                )
                .await
            {
                warn!("client_tool_call emit failed: {e}");
            }
        }
        ForwardAction::TaskStarted { id, task } => {
            let body = (id, task);
            if let Err(e) = connection
                .emit_signal::<&str, _, _, _, _>(
                    destination,
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
                    destination,
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
                    destination,
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
                    destination,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_broadcast_separates_per_user_stream_from_per_session_fanout() {
        // Per-user BROADCAST stream (rides the shared connection's
        // SubscribeBackgroundTasks): Task* + ConversationListChanged.
        assert!(
            ForwardAction::ConversationListChanged {
                conversation_id: "c".into()
            }
            .is_broadcast()
        );
        assert!(
            ForwardAction::TaskProgress {
                id: "t".into(),
                hint: String::new()
            }
            .is_broadcast()
        );
        assert!(
            ForwardAction::TaskCompleted {
                id: "t".into(),
                status: "completed".into(),
                last_error: String::new(),
            }
            .is_broadcast()
        );

        // Per-conversation fan-out (delivered to a per-sender session, UNICAST):
        // the response stream + UserMessageAdded must NOT be classed broadcast, or
        // `run_unicast` would drop them and a viewer would see no live turn.
        assert!(
            !ForwardAction::UserMessageAdded {
                conversation_id: "c".into(),
                request_id: "r".into(),
                content: "hi".into(),
            }
            .is_broadcast()
        );
        assert!(
            !ForwardAction::ResponseChunk {
                conversation_id: "c".into(),
                request_id: "r".into(),
                chunk: "x".into(),
            }
            .is_broadcast()
        );
        assert!(
            !ForwardAction::ResponseComplete {
                conversation_id: "c".into(),
                request_id: "r".into(),
                full_response: "x".into(),
            }
            .is_broadcast()
        );
        assert!(
            !ForwardAction::ResponseError {
                conversation_id: "c".into(),
                request_id: "r".into(),
                error: "x".into(),
            }
            .is_broadcast()
        );
        // #320: a tool call is unicast to the registrant — never broadcast (args
        // can be sensitive).
        assert!(
            !ForwardAction::ClientToolCall {
                task_id: "t".into(),
                conversation_id: "c".into(),
                tool_call_id: "tc".into(),
                tool_name: "echo".into(),
                arguments_json: "{}".into(),
            }
            .is_broadcast()
        );
    }
}
