//! A single high-level handle over any transport.
//!
//! [`Connector`] wraps [`connect_transport`](crate::transport::connect_transport):
//! it owns the chosen [`TransportClient`] *and* the daemon's signal stream,
//! fanning every [`SignalEvent`] out to any number of
//! [`subscribe`](Connector::subscribe)rs. A client issues commands and reads
//! events through one object instead of juggling a `(client, receiver)` pair and
//! per-transport channel wiring — the transport choice (D-Bus / local UDS /
//! WebSocket) lives entirely in the [`ConnectionConfig`].

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use desktop_assistant_api_model as api;
use tokio::sync::mpsc;

use crate::config::ConnectionConfig;
use crate::signal::SignalEvent;
use crate::timeouts::EVENT_STALL_TIMEOUT;
use crate::transport::{AssistantClient, TransportClient, connect_transport, transport_label};

type Subscribers = Arc<Mutex<Vec<mpsc::UnboundedSender<SignalEvent>>>>;

/// Pump a transport's signal stream out to subscribers.
///
/// Closing the upstream stream is terminal: every subscriber gets a
/// `Disconnected { reason: "signal stream closed" }` and the pump stops.
///
/// A *stall* (open but silent) is treated as a **per-turn** signal, not a reason
/// to tear the connection down (voice#49, reopened). The stall clock only runs
/// while at least one subscriber is attached — i.e. a turn is potentially in
/// flight: if no event arrives within `stall_timeout`, every *current*
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
/// (the reopened voice#49: send acks, turn completes, client gets nothing).
/// `stall_timeout` is a parameter so tests can drive a short window without
/// waiting the production minute-plus.
fn spawn_fanout_with_stall_timeout(
    mut signal_rx: mpsc::UnboundedReceiver<SignalEvent>,
    stall_timeout: Duration,
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
                        // No progress for a turn that had a waiter: unstick the
                        // current subscribers, but KEEP the pump alive so the
                        // next turn still streams.
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

/// A connected assistant client plus its automatically-managed event stream.
pub struct Connector {
    client: TransportClient,
    subscribers: Subscribers,
    label: String,
}

impl Connector {
    /// Connect over the transport named by `config` and start pumping the
    /// daemon's signal stream to subscribers. Uses the default
    /// [`EVENT_STALL_TIMEOUT`] for stall detection (#221).
    pub async fn connect(config: &ConnectionConfig) -> Result<Self> {
        Self::connect_with_stall_timeout(config, EVENT_STALL_TIMEOUT).await
    }

    /// Like [`connect`](Self::connect) but with an explicit event-stream stall
    /// window (#221). While a subscriber is attached (a turn may be in flight), a
    /// connection that stays open but emits no event for `stall_timeout` surfaces
    /// a terminal [`SignalEvent::Disconnected`] to the *current* subscribers — but
    /// the pump keeps running so later turns still stream (voice#49). An idle
    /// connection with no subscribers is never stalled out. Mainly for tests;
    /// production callers normally want the default via [`connect`](Self::connect).
    pub async fn connect_with_stall_timeout(
        config: &ConnectionConfig,
        stall_timeout: Duration,
    ) -> Result<Self> {
        let (client, signal_rx) = connect_transport(config).await?;
        Ok(Self {
            client,
            subscribers: spawn_fanout_with_stall_timeout(signal_rx, stall_timeout),
            label: transport_label(config),
        })
    }

    /// A fresh receiver for the daemon's signal stream. Every subscriber sees
    /// every event from the moment it subscribes; drop the receiver to
    /// unsubscribe. Subscribe before sending a prompt so no early chunk is lost.
    pub fn subscribe(&self) -> mpsc::UnboundedReceiver<SignalEvent> {
        register(&self.subscribers)
    }

    /// The underlying transport client — the full [`AssistantClient`] surface
    /// plus [`as_commands`](TransportClient::as_commands) for socket-only
    /// commands not modelled on the convenience methods below.
    pub fn client(&self) -> &TransportClient {
        &self.client
    }

    /// Human-readable description of the active connection (e.g. "Connected via
    /// local socket …").
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Create a conversation (works over every transport).
    pub async fn create_conversation(&self, title: &str) -> Result<String> {
        self.client.create_conversation(title).await
    }

    /// Send a prompt (works over every transport).
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
    /// Client tools ride the socket command channel, so this is supported only
    /// over the UDS/WS transports; the D-Bus transport has no command channel
    /// for it and returns an error.
    pub async fn register_client_tools(
        &self,
        tools: Vec<api::ClientToolRegistration>,
    ) -> Result<usize> {
        match self.client.as_commands() {
            Some(commands) => commands.register_client_tools(tools).await,
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
            request_id: "r".into(),
            chunk: "one".into(),
        })
        .unwrap();
        assert!(matches!(live.recv().await, Some(SignalEvent::Chunk { .. })));
        // After the dead subscriber is retained-out, only `live` remains.
        tx.send(SignalEvent::Chunk {
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
        let subs = spawn_fanout_with_stall_timeout(rx, Duration::from_millis(50));
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
        let subs = spawn_fanout_with_stall_timeout(rx, Duration::from_millis(80));
        let mut a = register(&subs);

        // Send three events spaced under the stall window; none should trip it.
        for i in 0..3 {
            tx.send(SignalEvent::Chunk {
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

    /// A stall unsticks the *current* subscribers (so a wedged turn errors out)
    /// but must NOT kill the pump: a fresh subscriber for the next turn still
    /// gets its events. This is the per-turn-vs-terminal distinction at the
    /// heart of the voice#49 fix.
    #[tokio::test]
    async fn fanout_stall_unsticks_waiters_but_keeps_serving_later_turns() {
        let (tx, rx) = mpsc::unbounded_channel::<SignalEvent>();
        let stall = Duration::from_millis(40);
        let subs = spawn_fanout_with_stall_timeout(rx, stall);

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
}
