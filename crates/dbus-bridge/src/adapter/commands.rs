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
//! ## Routing and why background-task subscriptions are rejected here
//!
//! Turn-driving (`SendMessage`) and session-pinned commands —
//! `SubscribeConversations` (#367) and `RegisterClientTools` / `ClientToolResult`
//! (#320) — route through the **caller's own per-sender daemon session**
//! ([`crate::session`]), keyed by the D-Bus sender on the message header. So a
//! turn's events come back on that session, a subscription's fan-out is isolated
//! to that caller, and a registered tool's `ClientToolCall` unicasts back to the
//! registrant. Stateless request/response uses the bridge's shared session.
//!
//! Only the background-task subscriptions are rejected, because they set up a
//! streaming push for the lifetime of a connection that a one-shot D-Bus method
//! cannot provide:
//!
//! - **`SubscribeBackgroundTasks` / `UnsubscribeBackgroundTasks`** — a one-shot
//!   D-Bus method cannot push the `Event::Task*` frame stream they request; the
//!   bridge holds its own background-task subscription open and re-emits them as
//!   `BackgroundTasks.*` signals instead (see `main.rs`). Rejected up front with
//!   `NotSupported` rather than silently mis-served.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::{fdo, interface};

use crate::session::SessionRegistry;
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
        // NOTE: `SubscribeConversations` (#367) and `RegisterClientTools` /
        // `ClientToolResult` (#320) are intentionally NOT rejected — they route to
        // the caller's per-sender session (see the module docs and
        // `crate::session::route`), so each caller's viewed-set + tool bucket are
        // isolated and the turn's `ClientToolCall` unicasts back to that caller.
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
    /// Per-sender daemon sessions (#367/#320). Wired in production via
    /// [`with_sessions`](Self::with_sessions); `None` in unit tests.
    sessions: Option<Arc<SessionRegistry>>,
}

impl<T: BridgeTransport + 'static> DbusCommandsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self {
            transport,
            sessions: None,
        }
    }

    /// Wire the per-sender session registry (production) so a turn-driving command
    /// arriving on this generic JSON channel routes through the *caller's* own
    /// session — identical to the typed Conversations methods. Without it (unit
    /// tests), everything uses the shared transport.
    pub fn with_sessions(mut self, sessions: Arc<SessionRegistry>) -> Self {
        self.sessions = Some(sessions);
        self
    }

    /// Parse, policy-check, and dispatch one serialized `api::Command`, returning
    /// the serialized `api::CommandResult`. The testable core of `send_command`;
    /// `caller` is the D-Bus sender's unique name (from the message header), used
    /// to route turn-driving commands to that caller's session.
    async fn run_command(&self, caller: Option<&str>, command_json: &str) -> fdo::Result<String> {
        let command: api::Command = serde_json::from_str(command_json)
            .map_err(|e| fdo::Error::InvalidArgs(format!("invalid command JSON: {e}")))?;

        if let Some(reason) = reject_reason(&command) {
            return Err(fdo::Error::NotSupported(reason.to_string()));
        }

        let result = match self.sessions.as_ref() {
            Some(registry) => registry
                .route(caller, command, self.transport.as_ref())
                .await
                .map_err(map_transport_err)?,
            None => self
                .transport
                .request(command)
                .await
                .map_err(map_transport_err)?,
        };

        serde_json::to_string(&result)
            .map_err(|e| fdo::Error::Failed(format!("serialize command result: {e}")))
    }

    /// Record `caller`'s client-context sharing decision (#782). The testable
    /// core of `set_share_client_context`; `caller` is the D-Bus sender's unique
    /// name, taken directly so the policy is exercisable without a bus.
    ///
    /// A caller with no bus name is refused rather than silently accepted: there
    /// is no session to attach the preference to, and a client that believes it
    /// declared a preference nothing recorded is exactly the failure this whole
    /// change exists to remove.
    pub fn declare_client_context_sharing(
        &self,
        caller: Option<&str>,
        enabled: bool,
    ) -> fdo::Result<()> {
        let sender = caller.ok_or_else(|| {
            fdo::Error::Failed(
                "declaring the client-context sharing preference requires a D-Bus sender \
                 (none on the message)"
                    .to_string(),
            )
        })?;
        // Refused for the same reason as a missing sender: an adapter with no
        // registry has nowhere to put the decision, and answering `Ok` would tell
        // the caller its preference was recorded when nothing recorded it.
        // Production always wires one.
        let registry = self.sessions.as_ref().ok_or_else(|| {
            fdo::Error::Failed(
                "this bridge has no per-sender session registry, so it cannot honour a \
                 client-context sharing preference"
                    .to_string(),
            )
        })?;
        registry.declare_client_context_sharing(sender, enabled);
        Ok(())
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
    async fn send_command(
        &self,
        #[zbus(header)] hdr: zbus::message::Header<'_>,
        command_json: &str,
    ) -> fdo::Result<String> {
        let caller = hdr.sender().map(|s| s.as_str());
        self.run_command(caller, command_json).await
    }

    /// Declare whether this caller shares its device context - the user's name,
    /// login, home directory, hostname, timezone and OS - with the assistant
    /// (#782). This is the D-Bus form of the preference the socket transports
    /// carry in their connect handshake.
    ///
    /// Call it before any session-scoped command. The bridge holds the daemon
    /// connection, so a D-Bus caller cannot put the preference on a handshake
    /// itself; the bridge records it against this caller's bus name and builds
    /// that caller's daemon session from it. Re-declaring the same value is free;
    /// changing it drops the caller's open session so the next one carries the
    /// new decision.
    ///
    /// **A caller that never calls this shares nothing.** The bridge does not
    /// treat silence as consent.
    async fn set_share_client_context(
        &self,
        #[zbus(header)] hdr: zbus::message::Header<'_>,
        enabled: bool,
    ) -> fdo::Result<()> {
        let caller = hdr.sender().map(|s| s.as_str());
        self.declare_client_context_sharing(caller, enabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records every command it sees and replies with a canned result (or a
    /// daemon error).
    struct FakeTransport {
        seen: Mutex<Vec<api::Command>>,
        reply: api::CommandResult,
        fail_with: Option<String>,
    }

    impl FakeTransport {
        fn replying(reply: api::CommandResult) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply,
                fail_with: None,
            })
        }

        fn failing(message: &str) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: api::CommandResult::Ack,
                fail_with: Some(message.to_string()),
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
    }

    fn adapter(transport: Arc<FakeTransport>) -> DbusCommandsAdapter<FakeTransport> {
        DbusCommandsAdapter::new(transport)
    }

    /// A factory the declaration tests must never reach: declaring a preference
    /// records it, it does not open a daemon session.
    struct NeverFactory;

    #[async_trait::async_trait]
    impl crate::session::SessionFactory for NeverFactory {
        async fn create(
            &self,
            _sender: &str,
            _share_client_context: bool,
        ) -> Result<crate::session::SenderSession, BridgeTransportError> {
            panic!("declaring a preference must not open a daemon session");
        }
    }

    #[tokio::test]
    async fn round_trips_request_and_response_json() {
        let reply = api::CommandResult::Conversations(vec![api::ConversationSummary {
            id: "c1".into(),
            title: "hello".into(),
            message_count: 3,
            updated_at: "2026-06-13T00:00:00Z".into(),
            archived: false,
            tags: vec![],
        }]);
        let transport = FakeTransport::replying(reply.clone());
        let adapter = adapter(Arc::clone(&transport));

        let request = serde_json::to_string(&api::Command::ListConversations {
            max_age_days: None,
            include_archived: false,
        })
        .unwrap();

        let raw = adapter.run_command(None, &request).await.unwrap();
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
                .run_command(None, &request)
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
    async fn subscribe_conversations_is_routed_not_rejected() {
        // #367: SubscribeConversations now routes to the caller's per-sender
        // session instead of being rejected. With no registry wired (unit test),
        // run_command falls back to the shared transport, so it reaches the daemon
        // rather than erroring — the regression guard for "we stopped rejecting it".
        let transport = FakeTransport::replying(api::CommandResult::Ack);
        let adapter = adapter(Arc::clone(&transport));
        let request = serde_json::to_string(&api::Command::SubscribeConversations {
            conversation_ids: vec!["c1".into(), "c2".into()],
        })
        .unwrap();

        adapter
            .run_command(Some(":1.10"), &request)
            .await
            .expect("SubscribeConversations must no longer be rejected over D-Bus");
        assert_eq!(
            transport.count(),
            1,
            "it must reach the daemon (be routed), not be rejected"
        );
        assert!(matches!(
            transport.last(),
            api::Command::SubscribeConversations { .. }
        ));
    }

    #[tokio::test]
    async fn client_tool_commands_are_routed_not_rejected() {
        // #320: RegisterClientTools + ClientToolResult now route to the caller's
        // per-sender session instead of being rejected. With no registry wired
        // (unit test) run_command falls back to the shared transport, so they
        // reach the daemon — the regression guard for "we stopped rejecting them".
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

            adapter
                .run_command(Some(":1.10"), &request)
                .await
                .expect("client-tool commands must no longer be rejected over D-Bus");
            assert_eq!(
                transport.count(),
                1,
                "it must reach the daemon (be routed), not be rejected"
            );
        }
    }

    #[tokio::test]
    async fn propagates_daemon_error_verbatim() {
        let transport = FakeTransport::failing("boom: connection refused");
        let adapter = adapter(Arc::clone(&transport));
        let request = serde_json::to_string(&api::Command::Ping).unwrap();

        let err = adapter
            .run_command(None, &request)
            .await
            .expect_err("a daemon error must surface as an fdo error");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert!(
            format!("{err}").contains("boom: connection refused"),
            "daemon message must be preserved: {err}"
        );
    }

    #[tokio::test]
    async fn declaring_client_context_sharing_records_it_for_the_caller() {
        // #782: the declaration is what the caller's per-sender daemon session is
        // then built from, so it must land against that caller's bus name.
        let transport = FakeTransport::replying(api::CommandResult::Ack);
        let registry = Arc::new(SessionRegistry::new(Arc::new(NeverFactory)));
        let adapter = adapter(transport).with_sessions(Arc::clone(&registry));

        adapter
            .declare_client_context_sharing(Some(":1.7"), true)
            .expect("a caller may declare its preference");

        assert!(registry.declared_client_context_sharing(":1.7"));
        assert!(
            !registry.declared_client_context_sharing(":1.9"),
            "a declaration must not apply to any other caller"
        );
    }

    #[tokio::test]
    async fn declaring_client_context_sharing_without_a_caller_is_rejected() {
        // With no bus sender there is no session to attach the preference to.
        // Accepting it silently would leave a caller believing it had opted in
        // (or out) when nothing recorded the choice.
        let transport = FakeTransport::replying(api::CommandResult::Ack);
        let registry = Arc::new(SessionRegistry::new(Arc::new(NeverFactory)));
        let adapter = adapter(transport).with_sessions(Arc::clone(&registry));

        let err = adapter
            .declare_client_context_sharing(None, true)
            .expect_err("a senderless declaration must be refused");
        assert!(matches!(err, fdo::Error::Failed(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn rejects_malformed_json_without_dispatch() {
        let transport = FakeTransport::replying(api::CommandResult::Ack);
        let adapter = adapter(Arc::clone(&transport));

        let err = adapter
            .run_command(None, "{not valid json")
            .await
            .expect_err("malformed command JSON must be rejected");
        assert!(matches!(err, fdo::Error::InvalidArgs(_)), "got {err:?}");
        assert_eq!(transport.count(), 0);
    }
}
