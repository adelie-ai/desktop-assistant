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

use anyhow::Result;
use tokio::sync::mpsc;

use crate::config::ConnectionConfig;
use crate::signal::SignalEvent;
use crate::transport::{AssistantClient, TransportClient, connect_transport, transport_label};

type Subscribers = Arc<Mutex<Vec<mpsc::UnboundedSender<SignalEvent>>>>;

/// Pump a transport's signal stream out to all registered subscribers until it
/// closes, then notify them with a final [`SignalEvent::Disconnected`].
fn spawn_fanout(mut signal_rx: mpsc::UnboundedReceiver<SignalEvent>) -> Subscribers {
    let subscribers: Subscribers = Arc::new(Mutex::new(Vec::new()));
    let pump = Arc::clone(&subscribers);
    tokio::spawn(async move {
        while let Some(event) = signal_rx.recv().await {
            // Deliver to every live subscriber; drop those whose receiver is gone.
            pump.lock()
                .unwrap()
                .retain(|tx| tx.send(event.clone()).is_ok());
        }
        // The transport closed — give subscribers a terminal event to react to.
        for tx in pump.lock().unwrap().drain(..) {
            let _ = tx.send(SignalEvent::Disconnected {
                reason: "signal stream closed".to_string(),
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
    /// daemon's signal stream to subscribers.
    pub async fn connect(config: &ConnectionConfig) -> Result<Self> {
        let (client, signal_rx) = connect_transport(config).await?;
        Ok(Self {
            client,
            subscribers: spawn_fanout(signal_rx),
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
