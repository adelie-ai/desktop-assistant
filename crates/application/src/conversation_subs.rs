//! Per-connection conversation subscriptions for live multi-client sync (#1).
//!
//! Each connection registers its outbound [`EventSink`] under its session id and
//! declares the set of conversations it is currently viewing. A turn then fans
//! its events to every OTHER connection viewing that conversation, so a turn
//! started by one client — or by the voice daemon — renders live in another
//! client that happens to be looking at the same conversation, instead of only
//! appearing after a reload.
//!
//! The originating connection is deliberately excluded from the fan-out: it
//! already receives the turn's events through its own per-request sink, so
//! routing to it as well would double-render. Keyed by session id (one per
//! connection), the registry stays correct when one account has several
//! connections open (gtk + tui + voice).
//!
//! Delivery is via the connection's existing [`EventSink`] (an mpsc behind the
//! transport writer), so it is reliable — not the lossy `Task*` broadcast — and
//! this module never needs to know about wire frames or the D-Bus bridge.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;

use crate::EventSink;

/// Shared registry of which connections are viewing which conversations, plus
/// each connection's sink, so turn events can be fanned to viewers.
#[derive(Default)]
pub struct ConversationSubscriptions {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// Live connections by session id → their outbound event sink.
    sinks: HashMap<String, Arc<dyn EventSink>>,
    /// Per-session set of subscribed conversation ids (what each connection is
    /// viewing). Set-replaced wholesale by `SubscribeConversations`.
    subscribed: HashMap<String, HashSet<String>>,
}

impl ConversationSubscriptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a connection's outbound sink, on connect. Idempotent.
    pub fn register(&self, session_id: &str, sink: Arc<dyn EventSink>) {
        let mut inner = self.lock();
        inner.sinks.insert(session_id.to_string(), sink);
    }

    /// Drop a connection on disconnect: forget its sink and its subscriptions so
    /// the maps stay bounded by live connections and no dead sink is routed to.
    pub fn unregister(&self, session_id: &str) {
        let mut inner = self.lock();
        inner.sinks.remove(session_id);
        inner.subscribed.remove(session_id);
    }

    /// Set-replace the conversations a connection is viewing. An empty list
    /// unsubscribes it from all (it still gets turns it initiates via its own
    /// request stream).
    pub fn set_subscriptions(&self, session_id: &str, conversation_ids: Vec<String>) {
        let mut inner = self.lock();
        inner.subscribed.insert(
            session_id.to_string(),
            conversation_ids.into_iter().collect(),
        );
    }

    /// Fan `event` (belonging to `conversation_id`) to every OTHER connection
    /// subscribed to that conversation. The origin is excluded — it receives the
    /// event via its own per-request sink. Best-effort: a sink whose connection
    /// has gone simply fails its emit and is cleaned up on disconnect.
    pub async fn route(&self, conversation_id: &str, event: &api::Event, origin_session: &str) {
        // Snapshot the target sinks under the lock, then release it before the
        // async emits so a slow/contended emit never holds the registry lock.
        let targets = self.subscribers_except(conversation_id, origin_session);
        for sink in targets {
            let _ = sink.emit(event.clone()).await;
        }
    }

    fn subscribers_except(
        &self,
        conversation_id: &str,
        origin_session: &str,
    ) -> Vec<Arc<dyn EventSink>> {
        let inner = self.lock();
        inner
            .subscribed
            .iter()
            .filter(|(session, convs)| {
                session.as_str() != origin_session && convs.contains(conversation_id)
            })
            .filter_map(|(session, _)| inner.sinks.get(session).cloned())
            .collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .expect("ConversationSubscriptions poisoned")
    }
}

/// The conversation a turn event belongs to, for fan-out routing. `None` for
/// events that are not part of a conversation's turn stream — those reach only
/// the originating connection.
fn turn_event_conversation_id(event: &api::Event) -> Option<&str> {
    match event {
        api::Event::UserMessageAdded {
            conversation_id, ..
        }
        | api::Event::AssistantDelta {
            conversation_id, ..
        }
        | api::Event::AssistantCompleted {
            conversation_id, ..
        }
        | api::Event::AssistantError {
            conversation_id, ..
        }
        | api::Event::AssistantStatus {
            conversation_id, ..
        } => Some(conversation_id),
        _ => None,
    }
}

/// A turn's event sink that delivers reliably to the originating connection
/// (`inner`) AND fans each turn event to every OTHER connection viewing the
/// same conversation (#1 live multi-client sync). Wrapping at the handler — the
/// chokepoint every transport's turn funnels through — means a turn driven over
/// ANY transport (a voice turn over D-Bus, a tui/gtk send over UDS/WS) fans to
/// viewers, without each transport re-implementing it. The fan-out is
/// best-effort and non-blocking (see [`FanoutTargetSink`]-style targets), so a
/// slow viewer never backpressures the origin's reliable delivery.
pub struct FanOutSink {
    inner: Arc<dyn EventSink>,
    subscriptions: Arc<ConversationSubscriptions>,
    origin_session: String,
}

impl FanOutSink {
    pub fn new(
        inner: Arc<dyn EventSink>,
        subscriptions: Arc<ConversationSubscriptions>,
        origin_session: String,
    ) -> Self {
        Self {
            inner,
            subscriptions,
            origin_session,
        }
    }
}

#[async_trait::async_trait]
impl EventSink for FanOutSink {
    async fn emit(&self, event: api::Event) -> bool {
        if let Some(conversation_id) = turn_event_conversation_id(&event) {
            self.subscriptions
                .route(conversation_id, &event, &self.origin_session)
                .await;
        }
        self.inner.emit(event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Records emitted events so a test can assert what a connection received.
    #[derive(Default)]
    struct RecordingSink(StdMutex<Vec<api::Event>>);

    #[async_trait::async_trait]
    impl EventSink for RecordingSink {
        async fn emit(&self, event: api::Event) -> bool {
            self.0.lock().unwrap().push(event);
            true
        }
    }

    fn delta(conv: &str) -> api::Event {
        api::Event::AssistantDelta {
            conversation_id: conv.to_string(),
            request_id: "r".into(),
            chunk: "hi".into(),
        }
    }

    #[tokio::test]
    async fn routes_to_other_subscribers_of_the_conversation() {
        let subs = ConversationSubscriptions::new();
        let viewer = Arc::new(RecordingSink::default());
        subs.register("viewer", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c1".into()]);

        subs.route("c1", &delta("c1"), "origin").await;

        assert_eq!(
            viewer.0.lock().unwrap().len(),
            1,
            "viewer of c1 must receive the turn event"
        );
    }

    #[tokio::test]
    async fn excludes_the_origin_connection() {
        let subs = ConversationSubscriptions::new();
        let origin = Arc::new(RecordingSink::default());
        subs.register("origin", origin.clone());
        subs.set_subscriptions("origin", vec!["c1".into()]);

        subs.route("c1", &delta("c1"), "origin").await;

        assert!(
            origin.0.lock().unwrap().is_empty(),
            "origin must NOT be fanned its own turn (it gets it via its own sink)"
        );
    }

    #[tokio::test]
    async fn does_not_route_to_subscribers_of_other_conversations() {
        let subs = ConversationSubscriptions::new();
        let viewer = Arc::new(RecordingSink::default());
        subs.register("viewer", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c2".into()]);

        subs.route("c1", &delta("c1"), "origin").await;

        assert!(
            viewer.0.lock().unwrap().is_empty(),
            "a connection viewing c2 must not receive c1's turn events"
        );
    }

    #[tokio::test]
    async fn set_replace_and_unregister_stop_delivery() {
        let subs = ConversationSubscriptions::new();
        let viewer = Arc::new(RecordingSink::default());
        subs.register("viewer", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c1".into()]);

        // Switch away from c1 (set-replace to a different set).
        subs.set_subscriptions("viewer", vec!["c2".into()]);
        subs.route("c1", &delta("c1"), "origin").await;
        assert!(
            viewer.0.lock().unwrap().is_empty(),
            "after switching away, no c1 delivery"
        );

        // Re-subscribe, then disconnect.
        subs.set_subscriptions("viewer", vec!["c1".into()]);
        subs.unregister("viewer");
        subs.route("c1", &delta("c1"), "origin").await;
        assert!(
            viewer.0.lock().unwrap().is_empty(),
            "after disconnect, no delivery"
        );
    }
}
