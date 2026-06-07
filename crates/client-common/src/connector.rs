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
use tokio::sync::mpsc;

use crate::config::ConnectionConfig;
use crate::signal::SignalEvent;
use crate::timeouts::EVENT_STALL_TIMEOUT;
use crate::transport::{AssistantClient, TransportClient, connect_transport, transport_label};

type Subscribers = Arc<Mutex<Vec<mpsc::UnboundedSender<SignalEvent>>>>;

/// Pump a transport's signal stream out to subscribers, surfacing both a closed
/// AND a *stalled* (open but silent) connection as a terminal
/// [`SignalEvent::Disconnected`] (#221).
///
/// Every received event resets the stall clock; if no event arrives within
/// `stall_timeout` the connection is assumed wedged and subscribers get a
/// `Disconnected { reason }` so a client waiting on the stream errors out
/// instead of hanging forever. This pairs with the orchestrator emitting
/// periodic `AssistantStatus`, which keeps a healthy connection's clock fresh
/// even between LLM chunks. `stall_timeout` is a parameter so tests can drive a
/// short window without waiting the production minute-plus.
fn spawn_fanout_with_stall_timeout(
    mut signal_rx: mpsc::UnboundedReceiver<SignalEvent>,
    stall_timeout: Duration,
) -> Subscribers {
    let subscribers: Subscribers = Arc::new(Mutex::new(Vec::new()));
    let pump = Arc::clone(&subscribers);
    tokio::spawn(async move {
        // The terminal reason depends on *why* we stopped: a closed upstream vs.
        // a stall the client should distinguish (and may choose to reconnect on).
        let reason = loop {
            match tokio::time::timeout(stall_timeout, signal_rx.recv()).await {
                Ok(Some(event)) => {
                    // Deliver to every live subscriber; drop those whose
                    // receiver is gone. Receipt also resets the stall clock,
                    // since the next `timeout` starts fresh on the next iter.
                    pump.lock()
                        .unwrap()
                        .retain(|tx| tx.send(event.clone()).is_ok());
                }
                Ok(None) => break "signal stream closed".to_string(),
                Err(_elapsed) => {
                    break format!(
                        "connection stalled: no events for {}s",
                        stall_timeout.as_secs()
                    );
                }
            }
        };
        // The transport closed or stalled — give subscribers a terminal event.
        for tx in pump.lock().unwrap().drain(..) {
            let _ = tx.send(SignalEvent::Disconnected {
                reason: reason.clone(),
            });
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
    /// window (#221). A connection that stays open but emits no event for
    /// `stall_timeout` surfaces a terminal [`SignalEvent::Disconnected`] to
    /// every subscriber. Mainly for tests; production callers normally want the
    /// default via [`connect`](Self::connect).
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
}
