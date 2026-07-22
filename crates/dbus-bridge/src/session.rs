//! Per-D-Bus-sender daemon sessions (#367 / #320).
//!
//! Before this, the bridge multiplexed **every** D-Bus caller onto one shared
//! daemon UDS session (`main.rs`'s startup [`Connector`]). That is fine for
//! stateless request/response, but it is exactly why live multi-client sync
//! (#367) and working client tools (#320) were rejected over D-Bus: both pin
//! per-*connection* state in the daemon (the `SubscribeConversations` viewed-set;
//! the client-tool bucket; the turn's event stream), and one shared connection
//! cannot keep one D-Bus caller's state from capturing or leaking to another's.
//!
//! This module gives each D-Bus sender its **own** daemon session: a private,
//! authenticated [`Connector`] (own daemon `session_id`, own minted JWT, own
//! reconnect supervisor) plus a per-session [unicast forwarder] that streams that
//! session's turn responses back to *only* that sender's unique bus name. A turn
//! a caller drives therefore behaves exactly as it would over UDS/WS — its events
//! come back on its own connection — instead of broadcasting across the bus.
//!
//! Sessions are created lazily, on a sender's first session-scoped call, and
//! evicted when that sender drops off the bus ([`spawn_name_owner_watcher`]);
//! dropping the [`SenderSession`] aborts its forwarder and drops its `Connector`,
//! which disconnects from the daemon — clearing that session's subscriptions and
//! tool registrations daemon-side.
//!
//! [unicast forwarder]: crate::adapter::event_forwarder::run_unicast

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;
use desktop_assistant_client_common::{ConnectionConfig, Connector};
use tracing::debug;
use zbus::Connection;

use crate::adapter::event_forwarder;
use crate::transport::{BridgeTransport, BridgeTransportError, ConnectorBridgeTransport};

/// Whether `command` drives a conversation turn and so must run on the caller's
/// **own** per-sender session — only then do its streamed `Assistant*` events
/// come back on that session (to be unicast to the caller), instead of being
/// lost on the shared connection. Everything else is fine on the shared
/// connection: stateless request/response returns its result inline, and
/// background-task commands are addressed by id, not by which connection owns
/// the turn. Cancelling a turn is `CancelBackgroundTask` (daemon-global by id),
/// so it is deliberately *not* here.
pub fn is_turn_driving(command: &api::Command) -> bool {
    matches!(command, api::Command::SendMessage { .. })
}

/// Whether `command` registers **per-connection** state that only has meaning on
/// the caller's own session: the `SubscribeConversations` live-sync viewed-set
/// (#367), and the client-tool registration + its result (#320) — a tool bucket
/// the daemon keys to the connection, whose `ClientToolCall` must come back on
/// that same connection (to be unicast to the caller). Unlike a turn (which can
/// fall back to the shared connection for a caller-less message), a session-pinned
/// command with no identifiable caller is rejected: routing it to the shared
/// connection would let one D-Bus caller's subscription or tools capture or leak
/// across every other caller (#270 / DT-4).
pub fn is_session_pinned(command: &api::Command) -> bool {
    matches!(
        command,
        api::Command::SubscribeConversations { .. }
            | api::Command::RegisterClientTools { .. }
            | api::Command::ClientToolResult { .. }
    )
}

/// One D-Bus sender's private daemon session: the transport its commands
/// dispatch through, plus the handle of the unicast forwarder pumping that
/// session's events to the sender. Dropping it aborts the forwarder (and, with
/// the forwarder gone and the transport's `Connector` Arc released, disconnects
/// from the daemon).
pub struct SenderSession {
    transport: Arc<dyn BridgeTransport>,
    forwarder: tokio::task::JoinHandle<()>,
}

impl SenderSession {
    /// Assemble a session from its transport and an already-spawned forwarder
    /// task. Production builds these in [`ConnectorSessionFactory`]; tests build
    /// them with a recording transport and a stub task.
    pub fn new(
        transport: Arc<dyn BridgeTransport>,
        forwarder: tokio::task::JoinHandle<()>,
    ) -> Self {
        Self {
            transport,
            forwarder,
        }
    }

    /// Dispatch `command` over this session's own daemon connection.
    pub async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        self.transport.request(command).await
    }
}

impl Drop for SenderSession {
    fn drop(&mut self) {
        // Stop the forwarder so an evicted session leaves no task pumping a dead
        // connection. The forwarder holds the only other `Connector` Arc, so
        // aborting it lets the `Connector` drop and disconnect from the daemon.
        self.forwarder.abort();
    }
}

/// Builds a fresh [`SenderSession`] for a sender. Abstracted so [`SessionRegistry`]
/// can be unit-tested without a live daemon or bus: production
/// ([`ConnectorSessionFactory`]) connects a real `Connector` and spawns the
/// unicast forwarder; tests return a recording transport and a stub task.
#[async_trait::async_trait]
pub trait SessionFactory: Send + Sync {
    /// Create a session whose forwarder unicasts to `sender` (a unique bus name).
    async fn create(&self, sender: &str) -> Result<SenderSession, BridgeTransportError>;
}

/// Production factory: each session is its own authenticated `Connector` to the
/// daemon plus a [`run_unicast`](event_forwarder::run_unicast) forwarder bound to
/// the sender's bus name.
///
/// The bus [`Connection`] (needed to emit the unicast signals) only exists after
/// the bridge binds its name, whereas the registry must be constructed *before*
/// that to be handed to the adapters. So the connection is supplied through a
/// shared slot filled immediately after `build()`. The slot is always populated
/// by the time `create` runs, because `create` is only reached from inside a
/// served D-Bus method — which cannot dispatch until the connection is up.
pub struct ConnectorSessionFactory {
    /// Template for each per-sender `Connector` (daemon + minter sockets, TTL).
    config: ConnectionConfig,
    /// The bridge's bus connection, filled right after the name is bound.
    connection: Arc<std::sync::OnceLock<Connection>>,
}

impl ConnectorSessionFactory {
    pub fn new(config: ConnectionConfig, connection: Arc<std::sync::OnceLock<Connection>>) -> Self {
        Self { config, connection }
    }
}

#[async_trait::async_trait]
impl SessionFactory for ConnectorSessionFactory {
    async fn create(&self, sender: &str) -> Result<SenderSession, BridgeTransportError> {
        let connection = self
            .connection
            .get()
            .ok_or_else(|| {
                BridgeTransportError::Daemon(
                    "bridge bus connection not initialised before a session was requested"
                        .to_string(),
                )
            })?
            .clone();

        // A private, authenticated daemon session for this sender: mints its own
        // JWT, handshakes, and owns reconnect + re-minting from here on.
        let connector = Arc::new(Connector::connect(&self.config).await.map_err(|e| {
            BridgeTransportError::Daemon(format!("failed to open a per-sender daemon session: {e}"))
        })?);
        let transport = Arc::new(ConnectorBridgeTransport::new(Arc::clone(&connector)));
        let forwarder = tokio::spawn(event_forwarder::run_unicast(
            connector,
            connection,
            sender.to_string(),
        ));
        Ok(SenderSession::new(transport, forwarder))
    }
}

/// Registry of live per-sender sessions, keyed by the caller's unique bus name.
pub struct SessionRegistry {
    factory: Arc<dyn SessionFactory>,
    sessions: Mutex<HashMap<String, Arc<SenderSession>>>,
}

impl SessionRegistry {
    pub fn new(factory: Arc<dyn SessionFactory>) -> Self {
        Self {
            factory,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Get the session for `sender`, creating it on first use. Idempotent: a
    /// sender that calls repeatedly reuses one daemon session.
    ///
    /// The async `create` runs outside the lock (so opening a connection never
    /// blocks other senders); if a concurrent first-call for the same sender wins
    /// the race, this call adopts the winner and drops its own freshly-built
    /// session (whose `Drop` disconnects it) — at most a wasted connect, never a
    /// duplicate in the map.
    pub async fn session_for(
        &self,
        sender: &str,
    ) -> Result<Arc<SenderSession>, BridgeTransportError> {
        if let Some(existing) = self.lock().get(sender).cloned() {
            return Ok(existing);
        }
        let session = Arc::new(self.factory.create(sender).await?);
        let mut sessions = self.lock();
        Ok(sessions
            .entry(sender.to_string())
            .or_insert(session)
            .clone())
    }

    /// Route `command` to the right daemon connection: the caller's own session
    /// when the command [drives a turn](is_turn_driving) and the caller is known,
    /// otherwise the shared `fallback` transport. This is the one place the
    /// shared-vs-per-sender decision lives, so both D-Bus surfaces (the typed
    /// Conversations methods and the generic Commands channel) route identically.
    pub async fn route(
        &self,
        caller: Option<&str>,
        command: api::Command,
        fallback: &dyn BridgeTransport,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        // Turn-driving: run on the caller's own session so its streamed events
        // come back on that session (to be unicast). A caller-less turn (no bus
        // sender — not expected on a real bus) falls back to the shared
        // connection, preserving prior behaviour.
        if is_turn_driving(&command)
            && let Some(sender) = caller
        {
            return self.session_for(sender).await?.request(command).await;
        }
        // Session-pinned: registers per-connection state that only has meaning on
        // the caller's own session. With no identifiable caller, reject rather
        // than fall back to the shared connection — a shared subscription would
        // capture or broadcast every caller's fan-out (#270).
        if is_session_pinned(&command) {
            let sender = caller.ok_or_else(|| {
                BridgeTransportError::Daemon(
                    "a session-scoped command requires a D-Bus sender (none on the message)"
                        .to_string(),
                )
            })?;
            return self.session_for(sender).await?.request(command).await;
        }
        fallback.request(command).await
    }

    /// Drop `sender`'s session if present, returning whether one was removed.
    /// Removing the last `Arc` runs [`SenderSession`]'s `Drop` (aborts the
    /// forwarder, disconnects the daemon session).
    pub fn evict(&self, sender: &str) -> bool {
        self.lock().remove(sender).is_some()
    }

    /// Number of live sessions (for the watcher's logging and tests).
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<SenderSession>>> {
        self.sessions.lock().expect("SessionRegistry poisoned")
    }
}

/// Apply one `NameOwnerChanged` to the registry: when a tracked sender's name has
/// **no new owner** (`new_owner` empty), it has dropped off the bus, so evict its
/// session. Factored out of [`spawn_name_owner_watcher`] so the eviction rule is
/// unit-testable without standing up a bus. Returns whether a session was evicted.
pub fn handle_name_owner_change(
    registry: &SessionRegistry,
    name: &str,
    new_owner: Option<&str>,
) -> bool {
    if new_owner.is_some() {
        return false; // name acquired or transferred, not a disconnect
    }
    let evicted = registry.evict(name);
    if evicted {
        debug!("evicted per-sender session for departed bus name {name}");
    }
    evicted
}

/// Watch `org.freedesktop.DBus` `NameOwnerChanged` and evict a sender's session
/// when it drops off the bus, so a crashed/exited D-Bus client can't leak its
/// daemon session (and, post-#320, can't wedge a turn on a tool call it will
/// never answer). Holds only a `Weak<SessionRegistry>` so the watcher cannot keep
/// the registry alive past shutdown.
pub fn spawn_name_owner_watcher(
    connection: Connection,
    registry: Arc<SessionRegistry>,
) -> tokio::task::JoinHandle<()> {
    use futures_util::StreamExt;
    let weak = Arc::downgrade(&registry);
    drop(registry);
    tokio::spawn(async move {
        let proxy = match zbus::fdo::DBusProxy::new(&connection).await {
            Ok(proxy) => proxy,
            Err(e) => {
                debug!(
                    "name-owner watcher: failed to build DBusProxy ({e}); sessions won't be GC'd on disconnect"
                );
                return;
            }
        };
        let mut stream = match proxy.receive_name_owner_changed().await {
            Ok(stream) => stream,
            Err(e) => {
                debug!("name-owner watcher: failed to subscribe NameOwnerChanged ({e})");
                return;
            }
        };
        while let Some(signal) = stream.next().await {
            let Ok(args) = signal.args() else { continue };
            let new_owner = args.new_owner().as_ref().map(|n| n.as_str());
            let Some(registry) = weak.upgrade() else {
                return; // registry gone — bridge shutting down
            };
            handle_name_owner_change(&registry, args.name().as_str(), new_owner);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Records every dispatched command so a test can assert isolation (which
    /// sender's session a command was routed to).
    #[derive(Default)]
    struct RecordingTransport {
        commands: StdMutex<Vec<api::Command>>,
    }

    #[async_trait::async_trait]
    impl BridgeTransport for RecordingTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.commands.lock().unwrap().push(command);
            Ok(api::CommandResult::Ack)
        }
    }

    /// Fake factory: hands back a recording transport per sender, records the
    /// order of `create` calls, and spawns a forever-pending forwarder so a test
    /// can prove eviction aborts it.
    #[derive(Default)]
    struct FakeFactory {
        created: StdMutex<Vec<String>>,
        transports: StdMutex<HashMap<String, Arc<RecordingTransport>>>,
        /// Optional per-sender forwarder probe: the spawned forwarder fires
        /// `started` once it is running (so a test can wait until the abort guard
        /// is installed, avoiding a spawn-vs-abort race) and `aborted` when its
        /// future is dropped on eviction.
        forwarder_probes: StdMutex<HashMap<String, ForwarderProbe>>,
        /// Optional barrier that parks `create` until N racers have arrived, so a
        /// test can deterministically force the concurrent-first-call path
        /// (create-outside-the-lock then `or_insert`) instead of a sequential
        /// create-then-reuse.
        gate: Option<Arc<tokio::sync::Barrier>>,
    }

    #[async_trait::async_trait]
    impl SessionFactory for FakeFactory {
        async fn create(&self, sender: &str) -> Result<SenderSession, BridgeTransportError> {
            // Park here until every racer has entered `create`, guaranteeing both
            // passed the empty-map check before either inserts.
            if let Some(gate) = &self.gate {
                gate.wait().await;
            }
            self.created.lock().unwrap().push(sender.to_string());
            let transport = Arc::new(RecordingTransport::default());
            self.transports
                .lock()
                .unwrap()
                .insert(sender.to_string(), Arc::clone(&transport));

            let probe = self.forwarder_probes.lock().unwrap().remove(sender);
            let forwarder = tokio::spawn(async move {
                // Signal "started", then install a guard that fires on abort
                // (when this future is dropped).
                let _aborted = probe.map(|p| {
                    let _ = p.started.send(());
                    SendOnDrop(Some(p.aborted))
                });
                std::future::pending::<()>().await;
            });
            Ok(SenderSession::new(transport, forwarder))
        }
    }

    /// Probes a fake forwarder's lifecycle for the eviction test.
    struct ForwarderProbe {
        started: tokio::sync::oneshot::Sender<()>,
        aborted: tokio::sync::oneshot::Sender<()>,
    }

    /// Fires its oneshot when dropped — lets a test observe a forwarder's future
    /// being dropped (on abort).
    struct SendOnDrop(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for SendOnDrop {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    fn registry_with(factory: Arc<FakeFactory>) -> SessionRegistry {
        SessionRegistry::new(factory)
    }

    #[tokio::test]
    async fn session_for_creates_once_then_reuses() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));

        let a1 = registry.session_for(":1.10").await.unwrap();
        let a2 = registry.session_for(":1.10").await.unwrap();

        assert_eq!(
            *factory.created.lock().unwrap(),
            vec![":1.10".to_string()],
            "a repeat caller must reuse one daemon session, not open a second"
        );
        assert!(
            Arc::ptr_eq(&a1, &a2),
            "same sender must get the same session"
        );
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn distinct_senders_get_distinct_sessions() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));

        let a = registry.session_for(":1.10").await.unwrap();
        let b = registry.session_for(":1.11").await.unwrap();

        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn routes_each_command_to_its_callers_own_session() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));

        registry
            .session_for(":1.10")
            .await
            .unwrap()
            .request(api::Command::Ping)
            .await
            .unwrap();
        registry
            .session_for(":1.11")
            .await
            .unwrap()
            .request(api::Command::GetStatus)
            .await
            .unwrap();

        let transports = factory.transports.lock().unwrap();
        let a_cmds = transports[":1.10"].commands.lock().unwrap().clone();
        let b_cmds = transports[":1.11"].commands.lock().unwrap().clone();
        assert!(
            matches!(a_cmds.as_slice(), [api::Command::Ping]),
            "sender A's command must land only on A's session, got {a_cmds:?}"
        );
        assert!(
            matches!(b_cmds.as_slice(), [api::Command::GetStatus]),
            "sender B's command must land only on B's session, got {b_cmds:?}"
        );
    }

    fn send_message(conv: &str) -> api::Command {
        api::Command::SendMessage {
            conversation_id: conv.to_string(),
            content: "hi".to_string(),
            override_selection: None,
            system_refinement: String::new(),
            client_context: None,
            idempotency_key: None,
        }
    }

    #[tokio::test]
    async fn route_sessions_turn_driving_commands_but_shares_everything_else() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));
        let fallback = Arc::new(RecordingTransport::default());

        // Turn-driving + known caller → the caller's own session, never the
        // shared connection (else the turn's events would be lost / broadcast).
        registry
            .route(Some(":1.10"), send_message("c1"), fallback.as_ref())
            .await
            .unwrap();
        assert!(
            fallback.commands.lock().unwrap().is_empty(),
            "a turn must not ride the shared connection"
        );
        {
            let transports = factory.transports.lock().unwrap();
            assert!(matches!(
                transports[":1.10"].commands.lock().unwrap().as_slice(),
                [api::Command::SendMessage { .. }]
            ));
        }

        // A non-turn command rides the shared fallback even with a known caller.
        registry
            .route(Some(":1.10"), api::Command::Ping, fallback.as_ref())
            .await
            .unwrap();
        assert!(matches!(
            fallback.commands.lock().unwrap().as_slice(),
            [api::Command::Ping]
        ));

        // Turn-driving but no identifiable caller → shared fallback (the bridge
        // can't pick a session), preserving today's behaviour for such callers.
        registry
            .route(None, send_message("c2"), fallback.as_ref())
            .await
            .unwrap();
        assert_eq!(
            fallback.commands.lock().unwrap().len(),
            2,
            "a caller-less turn falls back to the shared connection"
        );
    }

    fn subscribe(convs: &[&str]) -> api::Command {
        api::Command::SubscribeConversations {
            conversation_ids: convs.iter().map(|c| c.to_string()).collect(),
        }
    }

    #[test]
    fn is_session_pinned_covers_subscribe_and_client_tools() {
        assert!(is_session_pinned(&subscribe(&["c1"])));
        // #320: client-tool registration + its result are pinned to the session
        // that owns the tool bucket / the suspended turn.
        assert!(is_session_pinned(&api::Command::RegisterClientTools {
            tools: vec![]
        }));
        assert!(is_session_pinned(&api::Command::ClientToolResult {
            task_id: api::TaskId("t".into()),
            tool_call_id: "tc".into(),
            result: Some("ok".into()),
            error: None,
        }));
        // A turn is routed via is_turn_driving (which has a caller-less shared
        // fallback), NOT pinned; stateless commands are neither.
        assert!(!is_session_pinned(&send_message("c1")));
        assert!(!is_session_pinned(&api::Command::Ping));
    }

    #[tokio::test]
    async fn route_pins_session_scoped_commands_and_rejects_caller_less() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));
        let fallback = Arc::new(RecordingTransport::default());

        // SubscribeConversations with a known caller → that caller's own session,
        // never the shared connection (a shared sub would capture every caller's
        // fan-out — the #270 hazard).
        registry
            .route(Some(":1.10"), subscribe(&["c1"]), fallback.as_ref())
            .await
            .unwrap();
        assert!(
            fallback.commands.lock().unwrap().is_empty(),
            "a subscription must not ride the shared connection"
        );
        {
            let transports = factory.transports.lock().unwrap();
            assert!(matches!(
                transports[":1.10"].commands.lock().unwrap().as_slice(),
                [api::Command::SubscribeConversations { .. }]
            ));
        }

        // Caller-less session-pinned command → rejected, and crucially NOT routed
        // to the shared connection (which would broadcast its fan-out).
        let err = registry
            .route(None, subscribe(&["c2"]), fallback.as_ref())
            .await;
        assert!(
            err.is_err(),
            "a caller-less session-scoped command must be rejected"
        );
        assert!(
            fallback.commands.lock().unwrap().is_empty(),
            "and must not fall back to the shared connection"
        );
        assert_eq!(
            registry.len(),
            1,
            "the rejected caller-less command must add no session (only :1.10's remains)"
        );
    }

    #[tokio::test]
    async fn evict_removes_and_a_later_call_recreates() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));

        registry.session_for(":1.10").await.unwrap();
        assert!(registry.evict(":1.10"), "evict must report a removal");
        assert!(registry.is_empty());
        assert!(!registry.evict(":1.10"), "second evict is a no-op");

        registry.session_for(":1.10").await.unwrap();
        assert_eq!(
            *factory.created.lock().unwrap(),
            vec![":1.10".to_string(), ":1.10".to_string()],
            "a call after eviction must open a fresh session"
        );
    }

    #[tokio::test]
    async fn evicting_a_session_aborts_its_forwarder() {
        let factory = Arc::new(FakeFactory::default());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (aborted_tx, aborted_rx) = tokio::sync::oneshot::channel();
        factory.forwarder_probes.lock().unwrap().insert(
            ":1.10".to_string(),
            ForwarderProbe {
                started: started_tx,
                aborted: aborted_tx,
            },
        );
        let registry = registry_with(Arc::clone(&factory));

        registry.session_for(":1.10").await.unwrap();
        // Wait until the forwarder is actually running (abort guard installed), so
        // the eviction below can't race ahead of the task's first poll.
        tokio::time::timeout(std::time::Duration::from_secs(1), started_rx)
            .await
            .expect("forwarder should start")
            .expect("started signal");

        assert!(registry.evict(":1.10"));

        // Aborting the task drops its future, firing the guard.
        tokio::time::timeout(std::time::Duration::from_secs(1), aborted_rx)
            .await
            .expect("forwarder should be aborted promptly on eviction")
            .expect("aborted signal");
    }

    #[tokio::test]
    async fn name_owner_change_evicts_only_on_empty_new_owner() {
        let factory = Arc::new(FakeFactory::default());
        let registry = registry_with(Arc::clone(&factory));
        registry.session_for(":1.10").await.unwrap();

        // A new owner (name acquired/transferred) must NOT evict.
        assert!(!handle_name_owner_change(&registry, ":1.10", Some(":1.99")));
        assert_eq!(registry.len(), 1, "a live name must keep its session");

        // An untracked name dropping is a harmless no-op.
        assert!(!handle_name_owner_change(&registry, ":1.55", None));

        // The tracked name dropping (empty new owner) evicts.
        assert!(handle_name_owner_change(&registry, ":1.10", None));
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn concurrent_first_calls_for_one_sender_yield_one_session() {
        // Force both racers to be inside `create` at once (barrier of 2), so this
        // exercises the real create-outside-the-lock + `or_insert` path rather
        // than a sequential create-then-reuse.
        let factory = Arc::new(FakeFactory {
            gate: Some(Arc::new(tokio::sync::Barrier::new(2))),
            ..Default::default()
        });
        let registry = Arc::new(registry_with(Arc::clone(&factory)));

        let r1 = Arc::clone(&registry);
        let r2 = Arc::clone(&registry);
        let h1 = tokio::spawn(async move { r1.session_for(":1.10").await.unwrap() });
        let h2 = tokio::spawn(async move { r2.session_for(":1.10").await.unwrap() });
        let a = h1.await.unwrap();
        let b = h2.await.unwrap();

        assert!(
            Arc::ptr_eq(&a, &b),
            "racing first-calls for one sender must converge on a single session"
        );
        assert_eq!(
            registry.len(),
            1,
            "the loser's freshly-built session must not linger in the map"
        );
        assert_eq!(
            factory.created.lock().unwrap().len(),
            2,
            "both racers built a session (at most one wasted connect) — but only one is kept"
        );
    }

    #[test]
    fn is_turn_driving_is_send_message_only() {
        assert!(is_turn_driving(&send_message("c1")));
        // Stateless / id-addressed commands are fine on the shared connection.
        assert!(!is_turn_driving(&api::Command::Ping));
        assert!(!is_turn_driving(&api::Command::GetStatus));
        // Cancelling a turn is `CancelBackgroundTask` (daemon-global, by id), so
        // it deliberately does NOT need to run on the caller's session.
        assert!(!is_turn_driving(&api::Command::CancelBackgroundTask {
            id: "t".into()
        }));
    }

    /// A factory whose `create` always fails — to prove a turn whose session
    /// can't be opened surfaces the error instead of silently sharing.
    struct FailingFactory;
    #[async_trait::async_trait]
    impl SessionFactory for FailingFactory {
        async fn create(&self, _sender: &str) -> Result<SenderSession, BridgeTransportError> {
            Err(BridgeTransportError::Daemon("cannot open session".into()))
        }
    }

    #[tokio::test]
    async fn route_surfaces_a_session_creation_failure_without_falling_back() {
        let registry = SessionRegistry::new(Arc::new(FailingFactory));
        let fallback = Arc::new(RecordingTransport::default());

        let result = registry
            .route(Some(":1.10"), send_message("c1"), fallback.as_ref())
            .await;

        assert!(
            result.is_err(),
            "a turn whose session can't be opened must surface the error"
        );
        assert!(
            fallback.commands.lock().unwrap().is_empty(),
            "a failed session must NOT silently fall back to the shared connection — \
             that would broadcast the turn across the bus"
        );
        assert!(
            registry.is_empty(),
            "a failed create must leave no half-session behind"
        );
    }
}
