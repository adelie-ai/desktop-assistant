//! A single high-level handle over any transport.
//!
//! [`Connector`] wraps [`connect_transport`](crate::transport::connect_transport):
//! it owns the chosen [`TransportClient`] *and* the daemon's signal stream,
//! fanning every [`SignalEvent`] out to any number of
//! [`subscribe`](Connector::subscribe)rs. A client issues commands and reads
//! events through one object instead of juggling a `(client, receiver)` pair and
//! per-transport channel wiring — the transport choice (D-Bus / local UDS /
//! WebSocket) lives entirely in the [`ConnectionConfig`].
//!
//! ## Auto-reconnect (#246)
//!
//! The daemon is redeployed often (a binary restart), which closes every live
//! socket. Rather than strand the client on a dead connection until its own
//! process is restarted, the Connector runs a **supervisor** task that, when the
//! underlying socket *closes*, reconnects to the same endpoint with capped
//! exponential backoff + jitter — re-running the full handshake (re-auth via the
//! credential in the stored [`ConnectionConfig`]) and **replaying** the last
//! client-tool registration so a daemon restart doesn't silently drop a client's
//! tools. The current waiters are unstuck with a terminal `Disconnected` (their
//! in-flight turn is genuinely lost when the server restarts), but the Connector
//! stays usable: the next `subscribe()` / `send_prompt` / command runs on the
//! fresh transport.
//!
//! The reconnect happens *inside* the [`TransportClient`] (it swaps its own
//! socket in place and keeps feeding the same persistent signal stream), so
//! [`client`](Connector::client) keeps returning a stable `&TransportClient`
//! across reconnects — callers holding that reference don't have to re-fetch it.
//!
//! Reconnect is triggered **only** by an actual transport close (the transport's
//! drop-notifier fires), never by a per-turn *stall*: an open-but-silent
//! connection still unsticks its waiters (#221) and keeps the same connection
//! pumping for the next turn (voice#49/#241). An idle connection (no
//! subscribers, no traffic) is never torn down or reconnected.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use desktop_assistant_api_model as api;
use tokio::sync::mpsc;

use crate::config::{ConnectionConfig, TransportMode};
use crate::signal::SignalEvent;
use crate::timeouts::{
    DISPATCH_TIMEOUT, EVENT_STALL_TIMEOUT, RECONNECT_BACKOFF_INITIAL, RECONNECT_BACKOFF_MAX,
};
use crate::transport::{
    AssistantClient, DropNotifier, TransportClient, connect_transport, transport_label,
};

type Subscribers = Arc<Mutex<Vec<mpsc::UnboundedSender<SignalEvent>>>>;

/// The last client-tool registration the caller asked for, remembered so the
/// supervisor can replay it after a reconnect (#246). `None` until the first
/// [`Connector::register_client_tools`] call; an explicit empty `Vec` is
/// remembered too (it clears the daemon's set on every connect).
type RememberedTools = Arc<Mutex<Option<Vec<api::ClientToolRegistration>>>>;

/// Pump a transport's signal stream out to subscribers (#241 semantics).
///
/// Closing the upstream stream is terminal *for this pump*: it stops and every
/// subscriber gets a `Disconnected { reason: "signal stream closed" }`. With the
/// reconnect supervisor (#246) the transport's signal stream is *persistent* —
/// it does not close on a socket drop (the transport reconnects underneath and
/// keeps feeding it), so in practice this pump only ends when the Connector is
/// dropped. The terminal-on-close behaviour is retained for the D-Bus transport
/// (no reconnect) and as a safety net.
///
/// A *stall* (open but silent) is treated as a **per-turn** signal, not a reason
/// to tear the connection down (voice#49, reopened #241). The stall clock only
/// runs while at least one subscriber is attached — i.e. a turn is potentially
/// in flight: if no event arrives within `stall_timeout`, every *current*
/// subscriber gets a terminal `Disconnected { reason: "…stalled…" }` so a client
/// waiting on a wedged turn errors out instead of hanging (#221), but the pump
/// **keeps reading** so the next turn still streams.
///
/// While there are no subscribers, the connection is simply idle — the gap
/// between connecting and the first request, or between turns — and the pump
/// waits without a stall deadline. This is essential on transports without a
/// keepalive (UDS): otherwise a healthy-but-idle connection would trip the
/// stall and the pump would die before the first turn ever arrived, so every
/// later subscriber would be attached to a dead pump and receive ZERO events
/// (the reopened voice#49).
/// A liveness probe: resolves `true` if the connection is still alive. The
/// fanout uses it to tell an idle-but-alive link from a wedged/dead one before
/// declaring a stall — essential on a transport WITHOUT a keepalive (D-Bus),
/// where an idle connection produces no events and would otherwise trip the
/// stall window despite being healthy. `None` keeps the original blind stall (the
/// socket transports' pings already prevent false idle stalls).
type LivenessProbe = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

fn spawn_fanout_with_stall_timeout(
    mut signal_rx: mpsc::UnboundedReceiver<SignalEvent>,
    stall_timeout: Duration,
    liveness: Option<LivenessProbe>,
) -> Subscribers {
    let subscribers: Subscribers = Arc::new(Mutex::new(Vec::new()));
    let pump = Arc::clone(&subscribers);
    tokio::spawn(async move {
        loop {
            // The stall clock only applies while a subscriber is waiting (a turn
            // may be in flight). With no subscribers the connection is idle, not
            // wedged, so we wait unbounded — never stalling out a healthy link
            // that simply has no traffic yet (voice#49).
            let has_subscribers = !pump.lock().unwrap().is_empty();
            let next = if has_subscribers {
                match tokio::time::timeout(stall_timeout, signal_rx.recv()).await {
                    Ok(item) => item,
                    Err(_elapsed) => {
                        // Silent for the whole stall window with a waiter attached.
                        // With a liveness probe (D-Bus, which has no keepalive) a
                        // silent window usually just means the link is IDLE — so
                        // probe before tearing waiters off: alive ⇒ keep waiting,
                        // dead ⇒ disconnect. Without a probe (socket transports,
                        // whose pings keep events flowing) it's a genuine per-turn
                        // stall: unstick the current subscribers but KEEP the pump
                        // alive for the next turn (#221 / voice#49 / #246).
                        if let Some(probe) = &liveness
                            && probe().await
                        {
                            continue;
                        }
                        let reason = format!(
                            "connection stalled: no events for {}s",
                            stall_timeout.as_secs()
                        );
                        for tx in pump.lock().unwrap().drain(..) {
                            let _ = tx.send(SignalEvent::Disconnected {
                                reason: reason.clone(),
                            });
                        }
                        continue;
                    }
                }
            } else {
                signal_rx.recv().await
            };

            match next {
                Some(event) => {
                    // Deliver to every live subscriber; drop those whose receiver
                    // is gone. Receipt resets the stall clock implicitly — the
                    // next iteration re-arms the timeout from now.
                    pump.lock()
                        .unwrap()
                        .retain(|tx| tx.send(event.clone()).is_ok());
                }
                // Upstream closed — terminal. Notify whoever is left and stop.
                None => {
                    for tx in pump.lock().unwrap().drain(..) {
                        let _ = tx.send(SignalEvent::Disconnected {
                            reason: "signal stream closed".to_string(),
                        });
                    }
                    break;
                }
            }
        }
    });
    subscribers
}

fn register(subscribers: &Subscribers) -> mpsc::UnboundedReceiver<SignalEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    subscribers.lock().unwrap().push(tx);
    rx
}

/// Tell every current subscriber the connection dropped (so an in-flight turn
/// errors out instead of hanging) and clear the set. New events on the
/// reconnected stream go to whoever subscribes next.
fn disconnect_waiters(subscribers: &Subscribers, reason: &str) {
    for tx in subscribers.lock().unwrap().drain(..) {
        let _ = tx.send(SignalEvent::Disconnected {
            reason: reason.to_string(),
        });
    }
}

/// Next backoff delay: clamp `current` to the cap and add up to ~10% jitter so a
/// fleet of clients reconnecting after one daemon restart doesn't thunder in
/// lockstep. Jitter uses a cheap per-call varying source (sub-second nanos of
/// the wall clock); a tiny bias is harmless for a backoff. (The
/// `Math.random`/`Date::now` caveat is only for workflow scripts — fine here in
/// Rust.)
fn jittered_backoff(current: Duration) -> Duration {
    let base = current.min(RECONNECT_BACKOFF_MAX);
    let jitter_span = base.as_millis() as u64 / 10; // up to ~10%
    let jitter = if jitter_span == 0 {
        0
    } else {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        nanos % (jitter_span + 1)
    };
    base + Duration::from_millis(jitter)
}

/// Best-effort local hostname for the friendly tool-note host label (#248).
/// Dependency-free: the Linux kernel hostname, then `/etc/hostname`, then the
/// `HOSTNAME` env var. `None` when none resolve — the label is purely cosmetic,
/// so omitting it is fine (the daemon shows the generic "your device").
fn local_hostname() -> Option<String> {
    let from_file = |path: &str| {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    from_file("/proc/sys/kernel/hostname")
        .or_else(|| from_file("/etc/hostname"))
        .or_else(|| {
            std::env::var("HOSTNAME")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
}

/// Fill in the per-machine system id and host label on `config` for the connect
/// handshake (#248), unless the caller already set them (respected so tests can
/// inject a specific id). Reads the local id once (cached). Returns the
/// (possibly) updated config — stored by the Connector and re-read by the
/// reconnect supervisor, so the id rides both connect AND reconnect.
fn stamp_system_id(mut config: ConnectionConfig) -> ConnectionConfig {
    if config.system_id.is_none() {
        config.system_id = crate::system_id::local_system_id();
    }
    if config.host_label.is_none() {
        config.host_label = local_hostname();
    }
    config
}

/// The reconnect supervisor (#246). Waits on the transport's drop-notifier; on a
/// socket close it unsticks the current waiters and reconnects the *same*
/// `TransportClient` in place with capped exponential backoff + jitter,
/// replaying the remembered client-tool registration after each successful
/// reconnect. Runs until the task is aborted (Connector dropped) or the
/// notifier's sender is gone.
fn spawn_supervisor(
    config: ConnectionConfig,
    client: Arc<TransportClient>,
    subscribers: Subscribers,
    tools: RememberedTools,
    mut drop_rx: DropNotifier,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Each `()` is one underlying-socket close. The persistent signal stream
        // keeps flowing across reconnects, so the fan-out never sees the gap —
        // only this notifier does.
        while drop_rx.recv().await.is_some() {
            // The connection is gone. Unstick anyone still waiting on this turn
            // — it's lost when the server goes away — then reconnect.
            disconnect_waiters(&subscribers, "signal stream closed");

            // Reconnect with capped exponential backoff + jitter, retrying
            // indefinitely (a single attempt at a time — no thundering herd)
            // until the daemon comes back or this task is aborted.
            let mut backoff = RECONNECT_BACKOFF_INITIAL;
            loop {
                tokio::time::sleep(jittered_backoff(backoff)).await;
                match client.reconnect(&config).await {
                    Ok(()) => {
                        // Replay the last client-tool registration (#246) so a
                        // daemon restart doesn't silently lose this client's
                        // tools. The daemon replaces its set per call, so
                        // sending the full remembered list is correct. A failure
                        // here isn't fatal — the connection is up; log and carry
                        // on (the next explicit register, or the next reconnect,
                        // fixes it).
                        let remembered = tools.lock().unwrap().clone();
                        if let Some(tools_to_replay) = remembered
                            && let Some(commands) = client.as_commands()
                            && let Err(e) = commands.register_client_tools(tools_to_replay).await
                        {
                            tracing::warn!(
                                error = %e,
                                "failed to replay client-tool registration after reconnect"
                            );
                        }
                        tracing::info!("connector reconnected after transport drop");
                        break;
                    }
                    Err(e) => {
                        // Keep retrying with backoff — a long outage (or an
                        // expired JWT, acceptable for v1) backs off to the cap
                        // rather than spinning. Surface the reason at debug so an
                        // operator can see *why* it isn't reconnecting.
                        tracing::debug!(error = %e, "connector reconnect attempt failed; backing off");
                        backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
                    }
                }
            }
        }
    })
}

/// A connected assistant client plus its automatically-managed, auto-reconnecting
/// event stream.
pub struct Connector {
    client: Arc<TransportClient>,
    subscribers: Subscribers,
    /// Last client-tool registration, replayed after a reconnect (#246).
    tools: RememberedTools,
    label: String,
    /// The reconnect supervisor, if the transport supports reconnect (#246).
    /// `None` for D-Bus. Aborted on drop so the loop stops with the Connector.
    supervisor: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for Connector {
    fn drop(&mut self) {
        // Stop the reconnect loop when the last handle goes away; otherwise it
        // would keep trying to reconnect to a daemon no one is listening to.
        if let Some(handle) = &self.supervisor {
            handle.abort();
        }
    }
}

impl Connector {
    /// Connect over the transport named by `config` and start pumping the
    /// daemon's signal stream to subscribers, auto-reconnecting on a transport
    /// drop (#246). Uses the default [`EVENT_STALL_TIMEOUT`] for stall detection
    /// (#221).
    pub async fn connect(config: &ConnectionConfig) -> Result<Self> {
        Self::connect_with_stall_timeout(config, EVENT_STALL_TIMEOUT).await
    }

    /// Like [`connect`](Self::connect) but with an explicit event-stream stall
    /// window (#221). While a subscriber is attached (a turn may be in flight), a
    /// connection that stays open but emits no event for `stall_timeout` surfaces
    /// a terminal [`SignalEvent::Disconnected`] to the *current* subscribers — but
    /// the pump keeps running so later turns still stream (voice#49) and the
    /// transport is NOT torn down or reconnected (a stall is per-turn, not a
    /// close). An idle connection with no subscribers is never stalled out.
    /// Mainly for tests; production callers normally want the default via
    /// [`connect`](Self::connect).
    pub async fn connect_with_stall_timeout(
        config: &ConnectionConfig,
        stall_timeout: Duration,
    ) -> Result<Self> {
        // Stamp the per-machine system id (+ host label) onto the config so the
        // connect handshake carries it on EVERY transport (#248), and — because
        // the reconnect supervisor re-reads this same stored config — it is
        // re-sent on every reconnect too (#246/#247). A caller that already set
        // an id (e.g. a test) is respected; otherwise we read the local one. The
        // id is a co-location HINT, not a trust boundary — see `system_id`.
        let config = stamp_system_id(config.clone());

        // The initial connect is awaited (and may fail) so the caller gets a
        // clear error if the daemon isn't up at all; only *subsequent* drops
        // trigger the background reconnect loop.
        let (client, signal_rx, drop_rx) = connect_transport(&config).await?;
        let client = Arc::new(client);
        // D-Bus has no keepalive, so an idle connection emits no events and would
        // trip the stall window despite being healthy. Give the fanout a liveness
        // probe (a `Ping` round-trip) so it only stalls a D-Bus link that is
        // genuinely dead, not merely idle. Socket transports keep their pings, so
        // they get no probe and retain the original blind stall.
        let liveness: Option<LivenessProbe> = if config.transport_mode == TransportMode::Dbus {
            let probe_client = Arc::clone(&client);
            Some(Arc::new(move || {
                let probe_client = Arc::clone(&probe_client);
                Box::pin(async move {
                    match probe_client.as_commands() {
                        Some(cmds) => tokio::time::timeout(
                            DISPATCH_TIMEOUT,
                            cmds.send_command(api::Command::Ping),
                        )
                        .await
                        .is_ok_and(|r| r.is_ok()),
                        None => false,
                    }
                })
            }))
        } else {
            None
        };
        let subscribers = spawn_fanout_with_stall_timeout(signal_rx, stall_timeout, liveness);
        let tools: RememberedTools = Arc::new(Mutex::new(None));

        // Only the socket transports hand back a drop-notifier; D-Bus doesn't
        // reconnect (its clients don't use the Connector).
        let supervisor = drop_rx.map(|drop_rx| {
            spawn_supervisor(
                config.clone(),
                Arc::clone(&client),
                Arc::clone(&subscribers),
                Arc::clone(&tools),
                drop_rx,
            )
        });

        let label = transport_label(&config);
        Ok(Self {
            client,
            subscribers,
            tools,
            label,
            supervisor,
        })
    }

    /// A fresh receiver for the daemon's signal stream. Every subscriber sees
    /// every event from the moment it subscribes; drop the receiver to
    /// unsubscribe. Subscribe before sending a prompt so no early chunk is lost.
    ///
    /// Survives reconnects (#246): the underlying transport reconnects in place
    /// and keeps feeding the same stream, so a subscriber registered *after* a
    /// drop receives the new connection's events. A subscriber that was waiting
    /// across the drop gets a terminal `Disconnected` (its turn is lost) and
    /// should re-subscribe for the next turn.
    pub fn subscribe(&self) -> mpsc::UnboundedReceiver<SignalEvent> {
        register(&self.subscribers)
    }

    /// The underlying transport client — the full [`AssistantClient`] surface
    /// plus [`as_commands`](TransportClient::as_commands) for socket-only
    /// commands not modelled on the convenience methods below.
    ///
    /// Stable across reconnects (#246): the same `TransportClient` swaps its own
    /// socket in place, so a held `&TransportClient` keeps working after a
    /// daemon restart — callers don't need to re-fetch it.
    pub fn client(&self) -> &TransportClient {
        &self.client
    }

    /// Human-readable description of the active connection (e.g. "Connected via
    /// local socket …"). Stable across reconnects (the endpoint doesn't change).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Create a conversation (works over every transport).
    pub async fn create_conversation(&self, title: &str) -> Result<String> {
        self.client.create_conversation(title).await
    }

    /// Send a prompt (works over every transport).
    ///
    /// While the transport is mid-reconnect (the daemon is down), the underlying
    /// socket client rejects the command immediately ("connection closed") or,
    /// once a fresh connection is up, bounds the wait by the per-command
    /// dispatch timeout (#221) — so a send during an outage fails fast or
    /// succeeds on the reconnected transport, never hangs forever. After
    /// reconnect, commands work normally.
    pub async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        self.client.send_prompt(conversation_id, prompt).await
    }

    /// Send a prompt carrying a per-request system-prompt refinement. Socket
    /// transports (UDS / WS) pass it as a dedicated field; the D-Bus transport —
    /// whose high-level client has no refinement field — folds it into the
    /// prompt (the historical fallback), so the call works everywhere.
    pub async fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> Result<String> {
        if let Some(commands) = self.client.as_commands() {
            commands
                .send_prompt_with_system_refinement(conversation_id, prompt, system_refinement)
                .await
        } else {
            let composed = if system_refinement.trim().is_empty() {
                prompt.to_string()
            } else {
                format!("{system_refinement}\n\n{prompt}")
            };
            self.client.send_prompt(conversation_id, &composed).await
        }
    }

    /// Send a prompt with a per-request system-prompt refinement AND a
    /// client-supplied **idempotency key** (#204), so a retry after a dropped
    /// connection re-attaches to the live turn (or replays a completed reply)
    /// instead of re-running it. Socket transports (UDS / WS) carry both as
    /// dedicated fields; the D-Bus transport has neither, so it folds the
    /// refinement into the prompt and drops the key — a dropped D-Bus call
    /// isn't recoverable this way, so use a socket transport for idempotent
    /// retry. `idempotency_key = None` behaves like
    /// [`send_prompt_with_system_refinement`](Self::send_prompt_with_system_refinement).
    pub async fn send_prompt_with_system_refinement_idempotent(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
        idempotency_key: Option<String>,
    ) -> Result<String> {
        if let Some(commands) = self.client.as_commands() {
            commands
                .send_prompt_idempotent(
                    conversation_id,
                    prompt,
                    None,
                    system_refinement.to_string(),
                    idempotency_key,
                )
                .await
        } else {
            let composed = if system_refinement.trim().is_empty() {
                prompt.to_string()
            } else {
                format!("{system_refinement}\n\n{prompt}")
            };
            self.client.send_prompt(conversation_id, &composed).await
        }
    }

    /// Advertise the set of client-local MCP tools this connection can run
    /// (#107/#231). The daemon replaces any previously-registered set on each
    /// call — send the full list, not deltas — so re-register on every connect.
    /// Returns the count of tools the daemon accepted.
    ///
    /// The Connector **remembers** the last registration and replays it
    /// automatically after an auto-reconnect (#246), so a daemon restart doesn't
    /// silently drop this client's tools — callers don't have to re-register on
    /// every reconnect.
    ///
    /// Client tools ride the socket command channel, so this is supported only
    /// over the UDS/WS transports; the D-Bus transport has no command channel
    /// for it and returns an error.
    pub async fn register_client_tools(
        &self,
        tools: Vec<api::ClientToolRegistration>,
    ) -> Result<usize> {
        match self.client.as_commands() {
            Some(commands) => {
                let count = commands.register_client_tools(tools.clone()).await?;
                // Remember the *accepted* registration only after the daemon
                // confirms it, so a rejected payload isn't replayed on reconnect.
                *self.tools.lock().unwrap() = Some(tools);
                Ok(count)
            }
            None => Err(anyhow::anyhow!(
                "register_client_tools requires a socket transport (UDS or WS); \
                 the D-Bus transport does not support client tools"
            )),
        }
    }

    /// Deliver the outcome of a
    /// [`SignalEvent::ClientToolCall`](crate::SignalEvent::ClientToolCall) back
    /// to the daemon so the suspended turn can resume (#107/#231). Pass the
    /// `task_id` and `tool_call_id` from the event and exactly one of
    /// `result` / `error` (encoded as `Ok` / `Err`).
    ///
    /// Supported only over the socket transports (UDS/WS), like
    /// [`register_client_tools`](Self::register_client_tools).
    pub async fn submit_client_tool_result(
        &self,
        task_id: &str,
        tool_call_id: &str,
        result: Result<String, String>,
    ) -> Result<()> {
        match self.client.as_commands() {
            Some(commands) => {
                commands
                    .submit_client_tool_result(task_id, tool_call_id, result)
                    .await
            }
            None => Err(anyhow::anyhow!(
                "submit_client_tool_result requires a socket transport (UDS or WS); \
                 the D-Bus transport does not support client tools"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fan-out with the production stall window — the default `connect` path —
    /// for tests that exercise delivery/close, not the stall itself.
    fn spawn_fanout(rx: mpsc::UnboundedReceiver<SignalEvent>) -> Subscribers {
        spawn_fanout_with_stall_timeout(rx, EVENT_STALL_TIMEOUT, None)
    }

    #[tokio::test]
    async fn fanout_delivers_to_every_subscriber() {
        let (tx, rx) = mpsc::unbounded_channel();
        let subs = spawn_fanout(rx);
        let mut a = register(&subs);
        let mut b = register(&subs);

        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "hi".into(),
        })
        .unwrap();

        assert!(matches!(a.recv().await, Some(SignalEvent::Chunk { .. })));
        assert!(matches!(b.recv().await, Some(SignalEvent::Chunk { .. })));
    }

    #[tokio::test]
    async fn fanout_drops_a_gone_subscriber_and_keeps_serving_others() {
        let (tx, rx) = mpsc::unbounded_channel();
        let subs = spawn_fanout(rx);
        let gone = register(&subs);
        let mut live = register(&subs);
        drop(gone); // its receiver is dropped — must not break delivery to `live`

        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "one".into(),
        })
        .unwrap();
        assert!(matches!(live.recv().await, Some(SignalEvent::Chunk { .. })));
        // After the dead subscriber is retained-out, only `live` remains.
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "two".into(),
        })
        .unwrap();
        assert!(matches!(live.recv().await, Some(SignalEvent::Chunk { .. })));
    }

    #[tokio::test]
    async fn fanout_emits_disconnected_when_the_stream_closes() {
        let (tx, rx) = mpsc::unbounded_channel();
        let subs = spawn_fanout(rx);
        let mut a = register(&subs);
        drop(tx); // close the upstream signal stream

        assert!(matches!(
            a.recv().await,
            Some(SignalEvent::Disconnected { .. })
        ));
    }

    /// #221: a connection that stays *open but silent* (sender held, no events)
    /// must surface a terminal `Disconnected` once the stall window elapses —
    /// otherwise a subscriber waiting on `recv()` hangs forever.
    #[tokio::test]
    async fn fanout_emits_disconnected_on_stall() {
        // Keep `tx` alive for the whole test: the stream is OPEN, just silent.
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let subs = spawn_fanout_with_stall_timeout(rx, Duration::from_millis(50), None);
        let mut a = register(&subs);

        let event = tokio::time::timeout(Duration::from_secs(2), a.recv())
            .await
            .expect("stall must produce a terminal event, not hang");
        match event {
            Some(SignalEvent::Disconnected { reason }) => {
                assert!(
                    reason.contains("stalled"),
                    "stall reason should be distinguishable from a clean close, got: {reason}"
                );
            }
            other => panic!("expected SignalEvent::Disconnected on stall, got {other:?}"),
        }
        drop(tx);
    }

    /// A steady trickle of events under the stall window must keep the
    /// connection alive — the stall clock resets on every received event.
    #[tokio::test]
    async fn fanout_received_events_reset_the_stall_clock() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let subs = spawn_fanout_with_stall_timeout(rx, Duration::from_millis(80), None);
        let mut a = register(&subs);

        // Send three events spaced under the stall window; none should trip it.
        for i in 0..3 {
            tx.send(SignalEvent::Chunk {
                conversation_id: "c".into(),
                request_id: "r".into(),
                chunk: format!("c{i}"),
            })
            .unwrap();
            assert!(
                matches!(a.recv().await, Some(SignalEvent::Chunk { .. })),
                "live trickle must be delivered, not treated as a stall"
            );
            tokio::time::sleep(Duration::from_millis(40)).await;
        }
        // Still alive: no Disconnected yet (we've only ever waited < window).
        drop(tx);
    }

    /// voice#49 (reopened): a connection that sits IDLE with no subscribers past
    /// the stall window must NOT tear the pump down. A subscriber that registers
    /// for the first turn *after* that idle period must still receive its events.
    /// (UDS has no keepalive, so this idle gap always exceeds the window.)
    #[tokio::test]
    async fn fanout_idle_without_subscribers_does_not_stall_out() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let subs = spawn_fanout_with_stall_timeout(rx, stall, None);

        // Idle well past the stall window with NO subscribers (the
        // connect→first-turn gap). The pump must survive.
        tokio::time::sleep(stall * 4).await;

        // First turn: subscribe, then events arrive — they must be delivered.
        let mut a = register(&subs);
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "hi".into(),
        })
        .unwrap();

        let event = tokio::time::timeout(Duration::from_secs(2), a.recv())
            .await
            .expect("a turn after an idle period must still deliver, not be a dead pump");
        assert!(
            matches!(event, Some(SignalEvent::Chunk { .. })),
            "expected the first turn's chunk after idle, got {event:?}"
        );
        drop(tx);
    }

    /// With a liveness probe (the D-Bus path), a connection that is silent past
    /// the stall window but whose probe reports ALIVE must NOT be stalled out —
    /// an idle-but-healthy link keeps waiting instead of getting a spurious
    /// Disconnected. (This is the KDE-widget idle-stall fix.)
    #[tokio::test]
    async fn fanout_alive_probe_suppresses_idle_stall() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let probe: LivenessProbe = Arc::new(|| Box::pin(async { true }));
        let subs = spawn_fanout_with_stall_timeout(rx, stall, Some(probe));
        let mut a = register(&subs);

        // Several stall windows pass; an alive probe must suppress the stall, so
        // the subscriber receives nothing (no Disconnected) and keeps waiting.
        let res = tokio::time::timeout(stall * 4, a.recv()).await;
        assert!(
            res.is_err(),
            "an alive probe must suppress the idle stall (no Disconnected)"
        );

        // A real event still flows after the idle period.
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "hi".into(),
        })
        .unwrap();
        assert!(matches!(a.recv().await, Some(SignalEvent::Chunk { .. })));
        drop(tx);
    }

    /// A silent connection whose liveness probe reports DEAD is still stalled out
    /// — the waiter gets a terminal Disconnected instead of hanging (#221).
    #[tokio::test]
    async fn fanout_dead_probe_still_stalls() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let probe: LivenessProbe = Arc::new(|| Box::pin(async { false }));
        let subs = spawn_fanout_with_stall_timeout(rx, stall, Some(probe));
        let mut a = register(&subs);

        let event = tokio::time::timeout(Duration::from_secs(2), a.recv())
            .await
            .expect("a dead probe must still surface a terminal stall");
        assert!(
            matches!(event, Some(SignalEvent::Disconnected { ref reason }) if reason.contains("stalled")),
            "expected Disconnected(stalled) when the probe reports dead, got {event:?}"
        );
        drop(tx);
    }

    /// A stall unsticks the *current* subscribers (so a wedged turn errors out)
    /// but must NOT kill the pump: a fresh subscriber for the next turn still
    /// gets its events. This is the per-turn-vs-terminal distinction at the
    /// heart of the voice#49 fix.
    #[tokio::test]
    async fn fanout_stall_unsticks_waiters_but_keeps_serving_later_turns() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let subs = spawn_fanout_with_stall_timeout(rx, stall, None);

        // Turn 1: a subscriber waits on a wedged (silent) connection → gets a
        // terminal Disconnected once the stall fires.
        let mut first = register(&subs);
        let event = tokio::time::timeout(Duration::from_secs(2), first.recv())
            .await
            .expect("the waiting subscriber must be unstuck by the stall");
        assert!(
            matches!(event, Some(SignalEvent::Disconnected { ref reason }) if reason.contains("stalled")),
            "the waiter on a wedged turn should get a stall Disconnected, got {event:?}"
        );

        // Turn 2: a NEW subscriber on the SAME (still-open) connection must
        // still receive events — the pump survived the stall.
        let mut second = register(&subs);
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r2".into(),
            chunk: "next".into(),
        })
        .unwrap();
        let event = tokio::time::timeout(Duration::from_secs(2), second.recv())
            .await
            .expect("the next turn must still stream after a prior stall");
        assert!(
            matches!(event, Some(SignalEvent::Chunk { ref chunk, .. }) if chunk == "next"),
            "expected the next turn's chunk, got {event:?}"
        );
        drop(tx);
    }

    #[tokio::test]
    async fn jittered_backoff_never_exceeds_cap_plus_jitter() {
        // Capped at MAX; jitter adds at most ~10% on top.
        let d = jittered_backoff(Duration::from_secs(120));
        let ceiling = RECONNECT_BACKOFF_MAX + RECONNECT_BACKOFF_MAX / 10;
        assert!(
            d <= ceiling,
            "backoff {d:?} should be capped at MAX (+jitter) {ceiling:?}"
        );
        assert!(
            d >= RECONNECT_BACKOFF_MAX,
            "backoff should be at least the cap when input exceeds it"
        );
    }

    #[tokio::test]
    async fn jittered_backoff_grows_from_initial() {
        let d = jittered_backoff(RECONNECT_BACKOFF_INITIAL);
        assert!(
            d >= RECONNECT_BACKOFF_INITIAL,
            "backoff {d:?} should be at least the (non-jittered) initial delay"
        );
        assert!(
            d < RECONNECT_BACKOFF_INITIAL * 2,
            "initial backoff plus ~10% jitter should stay well under a doubling"
        );
    }
}
