//! Pump `WsFrame::Event`s from the transport into D-Bus signals.
//!
//! Subscribes to [`BridgeTransport::subscribe_events`] and dispatches
//! each event to the matching object path on a [`zbus::Connection`].
//! Translation mirrors the in-process daemon's signal vocabulary
//! one-for-one:
//!
//! | Wire event                              | D-Bus signal                                  |
//! | --------------------------------------- | --------------------------------------------- |
//! | `AssistantDelta`                        | `Conversations.ResponseChunk`                 |
//! | `AssistantCompleted`                    | `Conversations.ResponseComplete`              |
//! | `AssistantError`                        | `Conversations.ResponseError`                 |
//! | `ConfigChanged`                         | `Settings.ConfigChanged`                      |
//! | other (`Task*`, `AssistantStatus`, ...) | dropped — out of scope for #106               |
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

use super::paths;
use super::settings::{ConfigData, config_data_from_event};

const CONV_INTERFACE: &str = "org.desktopAssistant.Conversations";
const SETTINGS_INTERFACE: &str = "org.desktopAssistant.Settings";

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
        api::Event::TaskStarted { .. } => ForwardAction::Ignored {
            kind: "task_started",
        },
        api::Event::TaskProgress { .. } => ForwardAction::Ignored {
            kind: "task_progress",
        },
        api::Event::TaskLogAppended { .. } => ForwardAction::Ignored {
            kind: "task_log_appended",
        },
        api::Event::TaskCompleted { .. } => ForwardAction::Ignored {
            kind: "task_completed",
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
        ForwardAction::Ignored { kind } => {
            debug!("event forwarder ignoring kind={kind}");
        }
    }
}
