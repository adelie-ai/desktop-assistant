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
//! drop-notifier fires), never by a per-turn *stall*: a turn that goes silent
//! past the stall window is failed for its waiter with a per-turn `Error` (#221)
//! while the same connection keeps pumping for the next turn (voice#49/#241). An
//! idle connection — one with no turn in flight, whether or not a persistent
//! listener is subscribed — is never stalled out, torn down, or reconnected.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use desktop_assistant_api_model as api;
use tokio::sync::mpsc;

use crate::config::ConnectionConfig;
use crate::signal::SignalEvent;
use crate::timeouts::{EVENT_STALL_TIMEOUT, RECONNECT_BACKOFF_INITIAL, RECONNECT_BACKOFF_MAX};
use crate::transport::{
    AssistantClient, DropNotifier, TransportClient, connect_transport, transport_label,
};

type Subscribers = Arc<Mutex<Vec<mpsc::UnboundedSender<SignalEvent>>>>;

/// The last client-tool registration the caller asked for, remembered so the
/// supervisor can replay it after a reconnect (#246). `None` until the first
/// [`Connector::register_client_tools`] call; an explicit empty `Vec` is
/// remembered too (it clears the daemon's set on every connect).
type RememberedTools = Arc<Mutex<Option<Vec<api::ClientToolRegistration>>>>;

/// Pump a transport's signal stream out to subscribers (#241 semantics, refined
/// so an idle persistent subscriber is never stalled out).
///
/// Closing the upstream stream is terminal *for this pump*: it stops and every
/// subscriber gets a `Disconnected { reason: "signal stream closed" }`. With the
/// reconnect supervisor (#246) the transport's signal stream is *persistent* —
/// it does not close on a socket drop (the transport reconnects underneath and
/// keeps feeding it), so in practice this pump only ends when the Connector is
/// dropped. The terminal-on-close behaviour is retained for the D-Bus transport
/// (no reconnect) and as a safety net.
///
/// A *stall* (open but silent) is a **per-turn timeout**, never a connection
/// close (voice#49, reopened #241). The stall clock runs only while a turn is
/// genuinely **in flight** — the stream has delivered a turn event
/// (`UserMessageAdded` / `Chunk` / `Status` / `ContextUsage`) whose terminal
/// (`Complete` / `Error`) has not yet arrived (see [`track_turn_lifecycle`]). If
/// such a turn then emits nothing for `stall_timeout`, every in-flight turn is
/// failed with a synthetic `Error { request_id, … }` so a client waiting on a
/// wedged turn errors out instead of hanging (#221) — but the subscribers and
/// the pump are **kept**, so the connection is *not* torn down and the next turn
/// still streams.
///
/// Crucially, a connection with **no turn in flight is never stalled**, even
/// when a persistent listener is subscribed. The old code armed the stall on
/// mere subscriber presence as a proxy for "a turn may be in flight"; but a
/// persistently-subscribed GUI (adele-gtk / adele-tui) holds one subscription
/// for the whole session, so every idle gap longer than the window (UDS has no
/// keepalive) tripped a spurious `Disconnected` and bounced the client into a
/// full reconnect. Gating on an actual in-flight turn fixes that while still
/// catching a genuinely wedged turn. An idle pump (no in-flight turn) waits
/// unbounded, so a later subscriber is never attached to a dead pump (voice#49).
fn spawn_fanout_with_stall_timeout(
    mut signal_rx: mpsc::UnboundedReceiver<SignalEvent>,
    stall_timeout: Duration,
) -> Subscribers {
    let subscribers: Subscribers = Arc::new(Mutex::new(Vec::new()));
    let pump = Arc::clone(&subscribers);
    tokio::spawn(async move {
        // request_id -> conversation_id for turns whose terminal event has not
        // arrived yet. Non-empty == "a turn is in flight"; only then does the
        // stall clock run. Keeping the conversation_id lets a stalled turn be
        // reported as a per-turn `Error` on the right conversation.
        let mut in_flight: HashMap<String, String> = HashMap::new();
        loop {
            // Arm the stall only when a turn is actually in flight AND someone is
            // listening. An idle connection (no in-flight turn) waits unbounded,
            // so a persistently-subscribed but idle GUI is never stalled out.
            let arm = !in_flight.is_empty() && !pump.lock().unwrap().is_empty();
            let next = if arm {
                match tokio::time::timeout(stall_timeout, signal_rx.recv()).await {
                    Ok(item) => item,
                    Err(_elapsed) => {
                        // The in-flight turn(s) went silent past the window: fail
                        // each one for its waiter with a terminal `Error`, but
                        // KEEP the subscribers and the pump. A stall is a per-turn
                        // timeout, NOT a transport close — the connection stays up
                        // and the next turn still streams (#246/voice#49). Clients
                        // already treat `Error { request_id }` as "this turn
                        // failed", so no client-side change is needed.
                        let error = format!(
                            "no response from the daemon for {}s; the turn was abandoned",
                            stall_timeout.as_secs()
                        );
                        let stalled: Vec<(String, String)> = in_flight.drain().collect();
                        let mut subs = pump.lock().unwrap();
                        for (request_id, conversation_id) in stalled {
                            subs.retain(|tx| {
                                tx.send(SignalEvent::Error {
                                    conversation_id: conversation_id.clone(),
                                    request_id: request_id.clone(),
                                    error: error.clone(),
                                })
                                .is_ok()
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
                    track_turn_lifecycle(&mut in_flight, &event);
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

/// Update the in-flight-turn set from a freshly-received event, so
/// [`spawn_fanout_with_stall_timeout`] can run the stall clock only while a turn
/// is genuinely awaiting more events.
///
/// A turn is "in flight" from its first streamed event (`UserMessageAdded` /
/// `Chunk` / `Status` / `ContextUsage`, all keyed by `request_id`) until its
/// terminal `Complete` / `Error`. A `ClientToolCall` *parks* the turn pending a
/// client-local tool result — the daemon is legitimately quiet while the client
/// runs the tool, so the affected conversation's turns leave the in-flight set
/// (they must not be mistaken for a stall) and re-arm when the turn resumes
/// streaming. All other events (titles, task/scratchpad/knowledge signals,
/// disconnects) carry no turn and don't touch the set.
fn track_turn_lifecycle(in_flight: &mut HashMap<String, String>, event: &SignalEvent) {
    match event {
        SignalEvent::UserMessageAdded {
            request_id,
            conversation_id,
            ..
        }
        | SignalEvent::Chunk {
            request_id,
            conversation_id,
            ..
        }
        | SignalEvent::Status {
            request_id,
            conversation_id,
            ..
        }
        | SignalEvent::ContextUsage {
            request_id,
            conversation_id,
            ..
        } => {
            in_flight.insert(request_id.clone(), conversation_id.clone());
        }
        SignalEvent::Complete { request_id, .. } | SignalEvent::Error { request_id, .. } => {
            in_flight.remove(request_id);
        }
        SignalEvent::ClientToolCall {
            conversation_id, ..
        } => {
            // The turn is parked on the client; can't map task_id→request_id
            // here, so disarm every turn for this conversation. It re-arms on the
            // next streamed event once the turn resumes.
            in_flight.retain(|_, conv| conv != conversation_id);
        }
        _ => {}
    }
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
    /// window (#221). While a turn is **in flight** (the stream has delivered a
    /// turn event whose terminal hasn't arrived), a connection that then emits
    /// nothing for `stall_timeout` fails that turn for its waiter with a per-turn
    /// [`SignalEvent::Error`] — but the subscribers and the pump are kept, so
    /// later turns still stream (voice#49) and the transport is NOT torn down or
    /// reconnected (a stall is per-turn, not a close). A connection with no turn
    /// in flight is never stalled out, even with a persistent listener
    /// subscribed. Mainly for tests; production callers normally want the default
    /// via [`connect`](Self::connect).
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
        let subscribers = spawn_fanout_with_stall_timeout(signal_rx, stall_timeout);
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

    /// Create a conversation with tags. Socket transports carry the tags on the
    /// wire; the D-Bus transport (which has no tags field) silently drops them
    /// and creates a plain conversation — acceptable because D-Bus callers never
    /// pass non-empty tags today.
    pub async fn create_conversation_with_tags(&self, title: &str, tags: Vec<String>) -> Result<String> {
        if let Some(commands) = self.client.as_commands() {
            commands.create_conversation_with_tags(title, tags).await
        } else {
            self.client.create_conversation(title).await
        }
    }

    /// Archive a conversation (move it out of active without deleting).
    pub async fn archive_conversation(&self, id: &str) -> Result<()> {
        self.client.archive_conversation(id).await
    }

    /// Permanently delete a conversation and all its messages.
    pub async fn delete_conversation(&self, id: &str) -> Result<()> {
        self.client.delete_conversation(id).await
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
        spawn_fanout_with_stall_timeout(rx, EVENT_STALL_TIMEOUT)
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

    /// The bug this fix targets: a subscriber that is attached but has **no turn
    /// in flight** (a persistently-subscribed but idle GUI) must NOT be stalled
    /// out. The old fan-out armed the stall on mere subscriber presence, so an
    /// idle GUI was bounced with a spurious `Disconnected` every window.
    #[tokio::test]
    async fn fanout_idle_subscriber_with_no_turn_does_not_stall() {
        // Keep `tx` alive for the whole test: the stream is OPEN, just silent.
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let subs = spawn_fanout_with_stall_timeout(rx, stall);
        let mut a = register(&subs);

        // Wait well past the window with a subscriber attached but no turn in
        // flight: nothing must be delivered — no spurious stall/Disconnected.
        let res = tokio::time::timeout(stall * 4, a.recv()).await;
        assert!(
            res.is_err(),
            "an idle subscriber with no in-flight turn must not receive a stall event, got {res:?}"
        );
        drop(tx);
    }

    /// Once a turn reaches its terminal `Complete` it leaves the in-flight set,
    /// so a subsequent silent gap is just an idle connection and must NOT stall.
    #[tokio::test]
    async fn fanout_completed_turn_disarms_the_stall() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let subs = spawn_fanout_with_stall_timeout(rx, stall);
        let mut a = register(&subs);

        // Arm a turn, then complete it.
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "hi".into(),
        })
        .unwrap();
        assert!(matches!(a.recv().await, Some(SignalEvent::Chunk { .. })));
        tx.send(SignalEvent::Complete {
            conversation_id: "c".into(),
            request_id: "r".into(),
            full_response: "hi there".into(),
        })
        .unwrap();
        assert!(matches!(a.recv().await, Some(SignalEvent::Complete { .. })));

        // Turn done → idle → the silent gap must not stall the subscriber.
        let res = tokio::time::timeout(stall * 4, a.recv()).await;
        assert!(
            res.is_err(),
            "a connection idle after a completed turn must not stall, got {res:?}"
        );
        drop(tx);
    }

    /// A steady trickle of events under the stall window must keep the
    /// connection alive — the stall clock resets on every received event.
    #[tokio::test]
    async fn fanout_received_events_reset_the_stall_clock() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let subs = spawn_fanout_with_stall_timeout(rx, Duration::from_millis(80));
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
        let subs = spawn_fanout_with_stall_timeout(rx, stall);

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

    /// A turn that is in flight (armed by a streamed event) and then goes silent
    /// past the window is failed with a per-turn `Error` for its `request_id` —
    /// NOT a connection `Disconnected` — and the subscriber + pump are kept, so
    /// the SAME subscriber still receives the next turn. This is the per-turn
    /// timeout vs. transport-close distinction at the heart of the voice#49 fix.
    #[tokio::test]
    async fn fanout_stall_fails_the_in_flight_turn_but_keeps_the_connection() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let subs = spawn_fanout_with_stall_timeout(rx, stall);
        let mut a = register(&subs);

        // Arm a turn: one chunk for request "r1", then go silent.
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r1".into(),
            chunk: "partial".into(),
        })
        .unwrap();
        assert!(matches!(a.recv().await, Some(SignalEvent::Chunk { .. })));

        // The silent gap past the window fails the in-flight turn with an Error
        // keyed by its request_id, not a Disconnected.
        let event = tokio::time::timeout(Duration::from_secs(2), a.recv())
            .await
            .expect("a stalled in-flight turn must produce a terminal event");
        match event {
            Some(SignalEvent::Error { request_id, .. }) => assert_eq!(
                request_id, "r1",
                "the stalled turn's request_id must be the one failed"
            ),
            other => panic!("expected a per-turn Error on stall, got {other:?}"),
        }

        // The connection survived: the SAME subscriber (not drained) still gets
        // the next turn, and the now-idle connection doesn't stall again.
        tx.send(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r2".into(),
            chunk: "next".into(),
        })
        .unwrap();
        let event = tokio::time::timeout(Duration::from_secs(2), a.recv())
            .await
            .expect("the next turn must still stream after a prior stall");
        assert!(
            matches!(event, Some(SignalEvent::Chunk { ref chunk, .. }) if chunk == "next"),
            "expected the next turn's chunk on the same subscriber, got {event:?}"
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
