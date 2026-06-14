//! Generic D-Bus command channel for the bridge (#213 parity, #315 G1).
//!
//! The in-process `dbus-interface` daemon exposes `org.desktopAssistant.Commands`
//! — the single `SendCommand(s) -> s` channel that funnels every typed
//! [`api::Command`] through one JSON-in/JSON-out method, the way the WebSocket
//! and UDS transports do (`AssistantCommands::send_command`). Without it on the
//! bridge the cutover would regress `tui`/`gtk` launched with `--transport
//! dbus`, which route their entire management surface (config, per-conversation
//! model override, purposes, named connections, knowledge, scratchpad, …)
//! through this one method. This adapter restores that parity over the bridge's
//! UDS connection: `command_json` in, `api::CommandResult` JSON out.
//!
//! ## Why some commands are rejected here
//!
//! The bridge multiplexes **every** D-Bus caller onto its *single* daemon UDS
//! session (the one minted at startup, [`crate::transport`]). A handful of
//! commands assume a private, persistent, per-caller channel and would either
//! be undeliverable or leak across callers if forwarded blindly, so they are
//! rejected up front with `NotSupported` rather than silently mis-served:
//!
//! - **`SubscribeBackgroundTasks` / `UnsubscribeBackgroundTasks`** — set up a
//!   streaming push of `Event::Task*` frames for the lifetime of a connection.
//!   A one-shot D-Bus method cannot push events; the bridge holds its own
//!   background-task subscription open and re-emits them as
//!   `BackgroundTasks.*` signals instead (see `main.rs`). Mirrors the
//!   in-process adapter's rejection (`dbus-interface/src/commands.rs`).
//! - **`SubscribeConversations`** — the set-replace live-sync subscription
//!   (#356) registers the *connection's* sink against a set of conversations.
//!   Over the bridge that is the one shared UDS connection, so a single D-Bus
//!   caller's subscription would silently capture fan-out for every other
//!   D-Bus caller. Live multi-client sync over D-Bus needs per-caller routing
//!   the bridge does not model in v1 (it is a post-cutover follow-up, after
//!   the in-process surface is retired in #319); reject it for now.
//! - **`RegisterClientTools` / `ClientToolResult`** — the #270 / DT-4 hazard.
//!   Registrations would land in the bridge session's single tool bucket,
//!   shared across every D-Bus caller, and the resulting `Event::ClientToolCall`
//!   has no D-Bus signal to be delivered on, so the turn would wedge until the
//!   suspension timeout. Reject both, exactly as the tactical fix does for the
//!   in-process surface.
//!
//! A client that genuinely needs streaming subscriptions or working
//! client-side tools should use a socket transport (WS/UDS), which gives each
//! client its own session.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

/// Translate a transport error into the `fdo::Error` the D-Bus caller sees.
/// The daemon's own error message is surfaced verbatim so debugging stays one
/// hop from the source.
fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => fdo::Error::Failed(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

/// If `command` must not be forwarded over the bridge's shared session, return
/// the human-readable reason; otherwise `None`. Kept as a free function so the
/// policy is unit-testable without a live transport.
fn reject_reason(command: &api::Command) -> Option<&'static str> {
    match command {
        api::Command::SubscribeBackgroundTasks | api::Command::UnsubscribeBackgroundTasks => Some(
            "background-task streaming is not available over the generic D-Bus command channel \
             (a single request/response method cannot push events); the bridge re-emits Task* as \
             BackgroundTasks signals, or use a socket transport (WS/UDS) and \
             Command::SubscribeBackgroundTasks for the live stream",
        ),
        api::Command::SubscribeConversations { .. } => Some(
            "live conversation subscription is not available over the generic D-Bus command \
             channel: the bridge multiplexes every D-Bus caller onto one daemon session, so a \
             subscription would capture fan-out for all callers; use a socket transport (WS/UDS) \
             for live multi-client sync",
        ),
        api::Command::RegisterClientTools { .. } | api::Command::ClientToolResult { .. } => Some(
            "client-tool registration is not available over the generic D-Bus command channel: \
             the bridge shares one daemon session across all D-Bus callers, so a registration \
             would leak across callers, and the resulting tool-call events cannot be delivered \
             back over a one-shot method; use a socket transport (WS/UDS)",
        ),
        _ => None,
    }
}

/// D-Bus adapter exposing the shared command channel over the bridge.
///
/// Mirrors `dbus-interface`'s `DbusCommandsAdapter` method-for-method
/// (interface name, method name, `s -> s` signature) so the introspection
/// parity gate diffs empty; the difference is purely where the work happens —
/// here every command goes out as an [`api::Command`] over the UDS
/// [`BridgeTransport`] instead of into an in-process handler.
pub struct DbusCommandsAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusCommandsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }
}

#[interface(name = "org.desktopAssistant.Commands")]
impl<T: BridgeTransport + 'static> DbusCommandsAdapter<T> {
    /// Run one request/response [`api::Command`] and return its
    /// [`api::CommandResult`] as a JSON string.
    ///
    /// `command_json` is a serialized `api::Command`; the reply is a serialized
    /// `api::CommandResult`. Streaming / client-tool commands are rejected (see
    /// the module docs); everything else is forwarded over the bridge's
    /// authenticated UDS session, whose minted identity is the user scope —
    /// there is no `$USER` resolution here (contrast the in-process adapter),
    /// because the daemon derives the user from the bridge's JWT.
    async fn send_command(&self, command_json: &str) -> fdo::Result<String> {
        let command: api::Command = serde_json::from_str(command_json)
            .map_err(|e| fdo::Error::InvalidArgs(format!("invalid command JSON: {e}")))?;

        if let Some(reason) = reject_reason(&command) {
            return Err(fdo::Error::NotSupported(reason.to_string()));
        }

        let result = self
            .transport
            .request(command)
            .await
            .map_err(map_transport_err)?;

        serde_json::to_string(&result)
            .map_err(|e| fdo::Error::Failed(format!("serialize command result: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::sync::broadcast;

    /// Records every command it sees and replies with a canned result (or a
    /// daemon error). `subscribe_events` is unused by this adapter but required
    /// by the trait, so it returns a live-but-idle receiver.
    struct FakeTransport {
        seen: Mutex<Vec<api::Command>>,
        reply: api::CommandResult,
        fail_with: Option<String>,
        events_tx: broadcast::Sender<api::Event>,
    }

    impl FakeTransport {
        fn replying(reply: api::CommandResult) -> Arc<Self> {
            let (events_tx, _rx) = broadcast::channel(4);
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply,
                fail_with: None,
                events_tx,
            })
        }

        fn failing(message: &str) -> Arc<Self> {
            let (events_tx, _rx) = broadcast::channel(4);
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: api::CommandResult::Ack,
                fail_with: Some(message.to_string()),
                events_tx,
            })
        }

        fn count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }

        fn last(&self) -> api::Command {
            self.seen
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("no command")
        }
    }

    #[async_trait::async_trait]
    impl BridgeTransport for FakeTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.seen.lock().unwrap().push(command);
            match &self.fail_with {
                Some(msg) => Err(BridgeTransportError::Daemon(msg.clone())),
                None => Ok(self.reply.clone()),
            }
        }

        fn subscribe_events(&self) -> broadcast::Receiver<api::Event> {
            self.events_tx.subscribe()
        }
    }

    fn adapter(transport: Arc<FakeTransport>) -> DbusCommandsAdapter<FakeTransport> {
        DbusCommandsAdapter::new(transport)
    }

    #[tokio::test]
    async fn round_trips_request_and_response_json() {
        let reply = api::CommandResult::Conversations(vec![api::ConversationSummary {
            id: "c1".into(),
            title: "hello".into(),
            message_count: 3,
            updated_at: "2026-06-13T00:00:00Z".into(),
            archived: false,
        }]);
        let transport = FakeTransport::replying(reply.clone());
        let adapter = adapter(Arc::clone(&transport));

        let request = serde_json::to_string(&api::Command::ListConversations {
            max_age_days: None,
            include_archived: false,
        })
        .unwrap();

        let raw = adapter.send_command(&request).await.unwrap();
        let decoded: api::CommandResult = serde_json::from_str(&raw).unwrap();
        assert_eq!(decoded, reply);
        assert!(matches!(
            transport.last(),
            api::Command::ListConversations {
                include_archived: false,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn rejects_background_task_subscriptions_before_dispatch() {
        for cmd in [
            api::Command::SubscribeBackgroundTasks,
            api::Command::UnsubscribeBackgroundTasks,
        ] {
            let transport = FakeTransport::replying(api::CommandResult::Ack);
            let adapter = adapter(Arc::clone(&transport));
            let request = serde_json::to_string(&cmd).unwrap();

            let err = adapter
                .send_command(&request)
                .await
                .expect_err("streaming subscription must be rejected");
            assert!(matches!(err, fdo::Error::NotSupported(_)), "got {err:?}");
            assert_eq!(
                transport.count(),
                0,
                "a rejected command must not reach the daemon"
            );
        }
    }

    #[tokio::test]
    async fn rejects_conversation_subscription_before_dispatch() {
        let transport = FakeTransport::replying(api::CommandResult::Ack);
        let adapter = adapter(Arc::clone(&transport));
        let request = serde_json::to_string(&api::Command::SubscribeConversations {
            conversation_ids: vec!["c1".into(), "c2".into()],
        })
        .unwrap();

        let err = adapter
            .send_command(&request)
            .await
            .expect_err("conversation subscription must be rejected over D-Bus");
        assert!(matches!(err, fdo::Error::NotSupported(_)), "got {err:?}");
        assert!(format!("{err}").contains("live multi-client sync"));
        assert_eq!(transport.count(), 0);
    }

    #[tokio::test]
    async fn rejects_client_tool_registration_and_result_before_dispatch() {
        // The DT-4 / #270 hazard: never forward these onto the shared session.
        let register = api::Command::RegisterClientTools { tools: vec![] };
        let result = api::Command::ClientToolResult {
            task_id: api::TaskId("t1".into()),
            tool_call_id: "call-1".into(),
            result: Some("ok".into()),
            error: None,
        };
        for cmd in [register, result] {
            let transport = FakeTransport::replying(api::CommandResult::Ack);
            let adapter = adapter(Arc::clone(&transport));
            let request = serde_json::to_string(&cmd).unwrap();

            let err = adapter
                .send_command(&request)
                .await
                .expect_err("client-tool commands must be rejected over D-Bus");
            assert!(matches!(err, fdo::Error::NotSupported(_)), "got {err:?}");
            assert_eq!(transport.count(), 0);
        }
    }

    #[tokio::test]
    async fn propagates_daemon_error_verbatim() {
        let transport = FakeTransport::failing("boom: connection refused");
        let adapter = adapter(Arc::clone(&transport));
        let request = serde_json::to_string(&api::Command::Ping).unwrap();

        let err = adapter
            .send_command(&request)
            .await
            .expect_err("a daemon error must surface as an fdo error");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert!(
            format!("{err}").contains("boom: connection refused"),
            "daemon message must be preserved: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_malformed_json_without_dispatch() {
        let transport = FakeTransport::replying(api::CommandResult::Ack);
        let adapter = adapter(Arc::clone(&transport));

        let err = adapter
            .send_command("{not valid json")
            .await
            .expect_err("malformed command JSON must be rejected");
        assert!(matches!(err, fdo::Error::InvalidArgs(_)), "got {err:?}");
        assert_eq!(transport.count(), 0);
    }
}
