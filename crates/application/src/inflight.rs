//! In-flight `SendMessage` re-attach (#204 phase 2).
//!
//! Phase 1 (completed-dedup) replays a *finished* turn's reply. This is the
//! other half: a duplicate `SendMessage` carrying an `idempotency_key` whose
//! original is *still running in this process* (a transient connection blip,
//! no daemon restart) re-attaches to the live turn — replaying the chunks
//! emitted so far, then forwarding the rest live — instead of running a second
//! turn and double-processing an action.
//!
//! A live keyed turn emits through a [`TeeSink`]: events go to the original
//! caller's sink unchanged *and* into an [`InFlightTurn`], which buffers every
//! event for late re-attachers and broadcasts it to those already attached.
//! The [`InFlightRegistry`] indexes live turns by `(user_id,
//! conversation_id, idempotency_key)`; a turn registers when it starts and
//! removes itself when it ends.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use desktop_assistant_api_model as api;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::broadcast;

use crate::EventSink;

/// Broadcast capacity for live re-attach delivery. A re-attacher that falls
/// more than this many events behind drops the overflow (logged by tokio as a
/// lag), but the terminal `AssistantCompleted` carries the full reply text, so
/// even a lagging re-attacher ends with the complete answer.
const INFLIGHT_BROADCAST_CAP: usize = 256;

/// A single live turn's fan-out hub: an append-only event buffer plus a live
/// broadcast. Buffer-append and broadcast happen under one lock so a
/// re-attacher can snapshot the buffer and subscribe atomically — no event is
/// duplicated or dropped at the snapshot/subscribe boundary.
pub(crate) struct InFlightTurn {
    buffer: AsyncMutex<Vec<api::Event>>,
    tx: broadcast::Sender<api::Event>,
}

impl InFlightTurn {
    fn new() -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(INFLIGHT_BROADCAST_CAP);
        Arc::new(Self {
            buffer: AsyncMutex::new(Vec::new()),
            tx,
        })
    }

    /// Record + fan out one event. `Err` from `send` just means there are no
    /// live subscribers right now; the buffer still keeps the event for a
    /// future re-attacher.
    async fn emit_event(&self, event: api::Event) {
        let mut buffer = self.buffer.lock().await;
        buffer.push(event.clone());
        let _ = self.tx.send(event);
    }

    /// Atomically snapshot the events emitted so far and subscribe to the
    /// rest. Holding the buffer lock across the `subscribe()` call makes this
    /// mutually exclusive with [`Self::emit_event`], so the snapshot and the
    /// live subscription partition the event stream exactly.
    pub(crate) async fn snapshot_and_subscribe(
        &self,
    ) -> (Vec<api::Event>, broadcast::Receiver<api::Event>) {
        let buffer = self.buffer.lock().await;
        let rx = self.tx.subscribe();
        (buffer.clone(), rx)
    }
}

/// `EventSink` that mirrors a turn's events to the original caller's sink
/// (unchanged behaviour) *and* into the fan-out hub for re-attachers.
pub(crate) struct TeeSink {
    primary: Arc<dyn EventSink>,
    hub: Arc<InFlightTurn>,
}

impl TeeSink {
    pub(crate) fn new(primary: Arc<dyn EventSink>, hub: Arc<InFlightTurn>) -> Self {
        Self { primary, hub }
    }
}

#[async_trait]
impl EventSink for TeeSink {
    async fn emit(&self, event: api::Event) -> bool {
        // Feed the hub first so a re-attacher that joins right after can't miss
        // an event the primary already received.
        self.hub.emit_event(event.clone()).await;
        self.primary.emit(event).await
    }
}

/// Re-attach a sink to a live turn: replay the buffered events, then forward
/// live events until the turn ends (the hub's sender drops) or the sink
/// disconnects. Holds only a [`broadcast::Receiver`], never a strong reference
/// to the [`InFlightTurn`], so it never keeps the hub (and thus the broadcast
/// channel) alive past the turn it's following.
pub(crate) async fn forward_inflight(
    replay: Vec<api::Event>,
    mut rx: broadcast::Receiver<api::Event>,
    sink: Arc<dyn EventSink>,
) {
    for event in replay {
        if !sink.emit(event).await {
            return;
        }
    }
    loop {
        match rx.recv().await {
            Ok(event) => {
                if !sink.emit(event).await {
                    return;
                }
            }
            Err(broadcast::error::RecvError::Closed) => return,
            Err(broadcast::error::RecvError::Lagged(_)) => {
                // Slow re-attacher; skip the dropped deltas and keep going. The
                // terminal AssistantCompleted still carries the full reply.
            }
        }
    }
}

/// In-memory index of live keyed turns, by `(user_id, conversation_id,
/// idempotency_key)`. A duplicate keyed `SendMessage` that finds a live entry
/// re-attaches instead of running again.
#[derive(Default)]
pub(crate) struct InFlightRegistry {
    map: Mutex<HashMap<(String, String, String), Arc<InFlightTurn>>>,
}

impl InFlightRegistry {
    fn key(
        user_id: &str,
        conversation_id: &str,
        idempotency_key: &str,
    ) -> (String, String, String) {
        (
            user_id.to_string(),
            conversation_id.to_string(),
            idempotency_key.to_string(),
        )
    }

    /// Atomically claim a fresh live slot, returning its hub. Returns `None`
    /// when a live turn for this key already exists — the caller should
    /// re-attach via [`Self::get`] instead. The check-and-insert under one lock
    /// closes the concurrent same-key race.
    pub(crate) fn register(
        &self,
        user_id: &str,
        conversation_id: &str,
        idempotency_key: &str,
    ) -> Option<Arc<InFlightTurn>> {
        let mut map = self.map.lock().unwrap();
        let k = Self::key(user_id, conversation_id, idempotency_key);
        if map.contains_key(&k) {
            return None;
        }
        let turn = InFlightTurn::new();
        map.insert(k, Arc::clone(&turn));
        Some(turn)
    }

    /// The live turn for this key, if one is currently running.
    pub(crate) fn get(
        &self,
        user_id: &str,
        conversation_id: &str,
        idempotency_key: &str,
    ) -> Option<Arc<InFlightTurn>> {
        let map = self.map.lock().unwrap();
        map.get(&Self::key(user_id, conversation_id, idempotency_key))
            .map(Arc::clone)
    }

    /// Remove the live entry once the turn ends, so later same-key requests
    /// fall through to completed-dedup (or run fresh).
    pub(crate) fn remove(&self, user_id: &str, conversation_id: &str, idempotency_key: &str) {
        let mut map = self.map.lock().unwrap();
        map.remove(&Self::key(user_id, conversation_id, idempotency_key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Recording sink: collects emitted events, optionally a fixed number then
    /// reports disconnect.
    struct RecordingSink {
        events: StdMutex<Vec<api::Event>>,
    }
    impl RecordingSink {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                events: StdMutex::new(Vec::new()),
            })
        }
        fn delta_chunks(&self) -> Vec<String> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter_map(|e| match e {
                    api::Event::AssistantDelta { chunk, .. } => Some(chunk.clone()),
                    _ => None,
                })
                .collect()
        }
    }
    #[async_trait]
    impl EventSink for RecordingSink {
        async fn emit(&self, event: api::Event) -> bool {
            self.events.lock().unwrap().push(event);
            true
        }
    }

    fn delta(chunk: &str) -> api::Event {
        api::Event::AssistantDelta {
            conversation_id: "c1".into(),
            request_id: "r".into(),
            chunk: chunk.into(),
        }
    }

    #[tokio::test]
    async fn reattacher_gets_buffered_then_live_events() {
        let turn = InFlightTurn::new();
        // Two chunks emitted before anyone re-attaches.
        turn.emit_event(delta("a")).await;
        turn.emit_event(delta("b")).await;

        // A re-attacher snapshots (gets a, b) and subscribes.
        let (replay, rx) = turn.snapshot_and_subscribe().await;
        let sink = RecordingSink::new();
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let forward = tokio::spawn(forward_inflight(replay, rx, sink_dyn));

        // More chunks emitted after re-attach arrive live.
        turn.emit_event(delta("c")).await;
        turn.emit_event(delta("d")).await;

        // Dropping the turn closes the channel so the forward task finishes.
        drop(turn);
        forward.await.unwrap();

        assert_eq!(sink.delta_chunks(), vec!["a", "b", "c", "d"]);
    }

    #[tokio::test]
    async fn snapshot_and_subscribe_partitions_the_stream() {
        // No event is duplicated across the buffer snapshot and the live
        // subscription, nor dropped at the boundary.
        let turn = InFlightTurn::new();
        turn.emit_event(delta("x")).await;
        let (replay, rx) = turn.snapshot_and_subscribe().await;
        turn.emit_event(delta("y")).await;

        let sink = RecordingSink::new();
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let forward = tokio::spawn(forward_inflight(replay, rx, sink_dyn));
        drop(turn);
        forward.await.unwrap();

        assert_eq!(
            sink.delta_chunks(),
            vec!["x", "y"],
            "x via replay, y via live — exactly once each"
        );
    }

    #[test]
    fn register_is_an_atomic_claim() {
        let reg = InFlightRegistry::default();
        assert!(
            reg.register("u", "c", "k").is_some(),
            "first claim wins the slot"
        );
        assert!(
            reg.register("u", "c", "k").is_none(),
            "a second claim for a live key loses"
        );
        assert!(reg.get("u", "c", "k").is_some(), "the live turn is visible");

        // A different (user, conv, key) tuple is independent.
        assert!(reg.register("u", "c", "other").is_some());
        assert!(reg.register("other", "c", "k").is_some());

        reg.remove("u", "c", "k");
        assert!(reg.get("u", "c", "k").is_none(), "removed entries are gone");
        assert!(
            reg.register("u", "c", "k").is_some(),
            "a freed key can be claimed again"
        );
    }
}
