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

/// Observer notified whenever a session's subscribed set changes, with the
/// session id and its new set of conversation ids (empty when the session
/// disconnects). Installed via [`ConversationSubscriptions::set_change_observer`].
///
/// The daemon installs none; it exists for the web-UI BFF, which reaches the
/// daemon over one shared connection and must forward the *union* of its browser
/// sessions' subscriptions upstream so the daemon fans other clients' turns to
/// it (adele-web-ui#35). The observer runs synchronously and MUST NOT block or
/// re-enter the registry — the BFF's observer only recomputes a union and pushes
/// it onto a channel.
pub type SubscriptionChangeObserver = Arc<dyn Fn(&str, &[String]) + Send + Sync>;

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
    /// Per-session owning `user_id`. Fan-out only ever delivers a turn's
    /// events to sessions belonging to the SAME user as the turn's origin
    /// (#432) — subscribing to a conversation id is not authorization to
    /// receive its content. Without this, a session could subscribe to
    /// another user's conversation UUID and receive its `AssistantDelta` /
    /// `AssistantCompleted` (full message content).
    users: HashMap<String, String>,
    /// Optional change observer (adele-web-ui#35). `None` in the daemon — the
    /// feature is zero-cost until a consumer (the BFF) installs one.
    observer: Option<SubscriptionChangeObserver>,
}

impl ConversationSubscriptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a connection's outbound sink and owning `user_id`, on connect.
    /// Idempotent. The `user_id` scopes fan-out so a turn's events never reach
    /// a different user's connection (#432).
    pub fn register(&self, session_id: &str, user_id: &str, sink: Arc<dyn EventSink>) {
        let mut inner = self.lock();
        inner.sinks.insert(session_id.to_string(), sink);
        inner
            .users
            .insert(session_id.to_string(), user_id.to_string());
    }

    /// Install a [`SubscriptionChangeObserver`] (adele-web-ui#35). Install it
    /// before any connection registers so no change is missed; the daemon
    /// installs none. Replaces any previous observer.
    pub fn set_change_observer(&self, observer: SubscriptionChangeObserver) {
        self.lock().observer = Some(observer);
    }

    /// Drop a connection on disconnect: forget its sink, subscriptions, and
    /// user mapping so the maps stay bounded by live connections and no dead
    /// sink is routed to. Notifies the change observer (if any) that the session
    /// now views nothing, so a consumer can drop its contribution to the union.
    pub fn unregister(&self, session_id: &str) {
        let observer = {
            let mut inner = self.lock();
            inner.sinks.remove(session_id);
            inner.subscribed.remove(session_id);
            inner.users.remove(session_id);
            inner.observer.clone()
        };
        // Fire after releasing the lock: the observer must never re-enter the
        // registry, and holding the lock across it would risk a deadlock.
        if let Some(observer) = observer {
            observer(session_id, &[]);
        }
    }

    /// Set-replace the conversations a connection is viewing. An empty list
    /// unsubscribes it from all (it still gets turns it initiates via its own
    /// request stream). Notifies the change observer (if any) with the new set.
    pub fn set_subscriptions(&self, session_id: &str, conversation_ids: Vec<String>) {
        let observer = {
            let mut inner = self.lock();
            inner.subscribed.insert(
                session_id.to_string(),
                conversation_ids.iter().cloned().collect(),
            );
            inner.observer.clone()
        };
        // Fire after releasing the lock (see `unregister`).
        if let Some(observer) = observer {
            observer(session_id, &conversation_ids);
        }
    }

    /// Fan `event` (belonging to `conversation_id`) to every OTHER connection
    /// subscribed to that conversation AND owned by the same user as the turn's
    /// origin (`origin_user`). The origin is excluded — it receives the event
    /// via its own per-request sink. Best-effort: a sink whose connection has
    /// gone simply fails its emit and is cleaned up on disconnect.
    ///
    /// The `origin_user` filter is the authorization boundary (#432): a
    /// conversation is owned by exactly one user, and the origin runs the turn
    /// under its own user scope on its own conversation, so "same user as the
    /// origin" is precisely "allowed to see this conversation". A session that
    /// merely subscribed to a foreign conversation UUID is not delivered to.
    pub async fn route(
        &self,
        conversation_id: &str,
        event: &api::Event,
        origin_session: &str,
        origin_user: &str,
    ) {
        // Snapshot the target sinks under the lock, then release it before the
        // async emits so a slow/contended emit never holds the registry lock.
        let targets = self.subscribers_except(conversation_id, origin_session, origin_user);
        for sink in targets {
            let _ = sink.emit(event.clone()).await;
        }
    }

    fn subscribers_except(
        &self,
        conversation_id: &str,
        origin_session: &str,
        origin_user: &str,
    ) -> Vec<Arc<dyn EventSink>> {
        let inner = self.lock();
        inner
            .subscribed
            .iter()
            .filter(|(session, convs)| {
                session.as_str() != origin_session
                    && convs.contains(conversation_id)
                    // #432: only deliver to sessions owned by the same user as
                    // the origin. A subscription is not authorization.
                    && inner.users.get(*session).map(String::as_str) == Some(origin_user)
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
    origin_user: String,
}

impl FanOutSink {
    pub fn new(
        inner: Arc<dyn EventSink>,
        subscriptions: Arc<ConversationSubscriptions>,
        origin_session: String,
        origin_user: String,
    ) -> Self {
        Self {
            inner,
            subscriptions,
            origin_session,
            origin_user,
        }
    }
}

#[async_trait::async_trait]
impl EventSink for FanOutSink {
    async fn emit(&self, event: api::Event) -> bool {
        if let Some(conversation_id) = turn_event_conversation_id(&event) {
            self.subscriptions
                .route(
                    conversation_id,
                    &event,
                    &self.origin_session,
                    &self.origin_user,
                )
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
        subs.register("viewer", "alice", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c1".into()]);

        subs.route("c1", &delta("c1"), "origin", "alice").await;

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
        subs.register("origin", "alice", origin.clone());
        subs.set_subscriptions("origin", vec!["c1".into()]);

        subs.route("c1", &delta("c1"), "origin", "alice").await;

        assert!(
            origin.0.lock().unwrap().is_empty(),
            "origin must NOT be fanned its own turn (it gets it via its own sink)"
        );
    }

    #[tokio::test]
    async fn does_not_route_to_subscribers_of_other_conversations() {
        let subs = ConversationSubscriptions::new();
        let viewer = Arc::new(RecordingSink::default());
        subs.register("viewer", "alice", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c2".into()]);

        subs.route("c1", &delta("c1"), "origin", "alice").await;

        assert!(
            viewer.0.lock().unwrap().is_empty(),
            "a connection viewing c2 must not receive c1's turn events"
        );
    }

    #[tokio::test]
    async fn does_not_route_to_a_different_users_subscription() {
        // #432: bob's session subscribes to alice's conversation id. When
        // alice runs a turn on it, bob must receive nothing — subscribing to a
        // conversation UUID is not authorization to see its content. Before the
        // fix, only UUID-unguessability stood between bob and alice's replies.
        let subs = ConversationSubscriptions::new();
        let bob = Arc::new(RecordingSink::default());
        subs.register("bob-sess", "bob", bob.clone());
        subs.set_subscriptions("bob-sess", vec!["alice-conv".into()]);

        // alice is the origin — she owns the conversation and runs the turn.
        subs.route("alice-conv", &delta("alice-conv"), "alice-sess", "alice")
            .await;

        assert!(
            bob.0.lock().unwrap().is_empty(),
            "a different user's subscription must not receive the turn's content"
        );
    }

    #[tokio::test]
    async fn routes_to_same_user_other_connection() {
        // alice viewing the same conversation from a second connection (e.g.
        // gtk while the turn runs from voice) still gets live events — the
        // user-scoping must not break legitimate same-user multi-client sync.
        let subs = ConversationSubscriptions::new();
        let gtk = Arc::new(RecordingSink::default());
        subs.register("alice-gtk", "alice", gtk.clone());
        subs.set_subscriptions("alice-gtk", vec!["c1".into()]);

        subs.route("c1", &delta("c1"), "alice-voice", "alice").await;

        assert_eq!(
            gtk.0.lock().unwrap().len(),
            1,
            "alice's other connection must still receive live sync"
        );
    }

    #[tokio::test]
    async fn set_replace_and_unregister_stop_delivery() {
        let subs = ConversationSubscriptions::new();
        let viewer = Arc::new(RecordingSink::default());
        subs.register("viewer", "alice", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c1".into()]);

        // Switch away from c1 (set-replace to a different set).
        subs.set_subscriptions("viewer", vec!["c2".into()]);
        subs.route("c1", &delta("c1"), "origin", "alice").await;
        assert!(
            viewer.0.lock().unwrap().is_empty(),
            "after switching away, no c1 delivery"
        );

        // Re-subscribe, then disconnect.
        subs.set_subscriptions("viewer", vec!["c1".into()]);
        subs.unregister("viewer");
        subs.route("c1", &delta("c1"), "origin", "alice").await;
        assert!(
            viewer.0.lock().unwrap().is_empty(),
            "after disconnect, no delivery"
        );
    }

    // --- Change observer (adele-web-ui#35) ------------------------------------
    //
    // The BFF forwards the union of its browser sessions' subscriptions onto its
    // single upstream daemon connection. It cannot see the dispatcher's
    // per-session `set_subscriptions` / `unregister` calls, so the registry
    // notifies an installed observer of each change. The daemon installs none, so
    // the feature is zero-cost by default.

    /// Records `(session_id, conversation_ids)` from each observer notification.
    #[derive(Default)]
    struct RecordingObserver(StdMutex<Vec<(String, Vec<String>)>>);

    impl RecordingObserver {
        fn calls(&self) -> Vec<(String, Vec<String>)> {
            self.0.lock().unwrap().clone()
        }
    }

    fn install_recording_observer(subs: &ConversationSubscriptions) -> Arc<RecordingObserver> {
        let rec = Arc::new(RecordingObserver::default());
        let sink = Arc::clone(&rec);
        subs.set_change_observer(Arc::new(move |session: &str, convs: &[String]| {
            sink.0
                .lock()
                .unwrap()
                .push((session.to_string(), convs.to_vec()));
        }));
        rec
    }

    #[test]
    fn change_observer_fires_with_the_new_set_on_set_subscriptions() {
        let subs = ConversationSubscriptions::new();
        let rec = install_recording_observer(&subs);

        subs.set_subscriptions("s1", vec!["c1".into(), "c2".into()]);

        assert_eq!(
            rec.calls(),
            vec![("s1".to_string(), vec!["c1".to_string(), "c2".to_string()])],
            "observer must be notified of the session's new subscribed set"
        );
    }

    #[test]
    fn change_observer_fires_empty_set_on_unregister() {
        let subs = ConversationSubscriptions::new();
        subs.register("s1", "alice", Arc::new(RecordingSink::default()));
        subs.set_subscriptions("s1", vec!["c1".into()]);
        let rec = install_recording_observer(&subs);

        subs.unregister("s1");

        assert_eq!(
            rec.calls(),
            vec![("s1".to_string(), Vec::<String>::new())],
            "a disconnect must be reported as the session viewing nothing"
        );
    }

    #[test]
    fn register_alone_does_not_fire_the_change_observer() {
        // Registering a connection's sink does not change what anyone is viewing
        // (its subscribed set is still empty), so the union is unchanged and the
        // observer must not fire until a `SubscribeConversations` arrives.
        let subs = ConversationSubscriptions::new();
        let rec = install_recording_observer(&subs);

        subs.register("s1", "alice", Arc::new(RecordingSink::default()));

        assert!(
            rec.calls().is_empty(),
            "register must not notify the change observer"
        );
    }

    #[tokio::test]
    async fn without_a_change_observer_mutations_and_routing_still_work() {
        // Degradation: the daemon installs no observer. Every mutation must be a
        // no-op notification-wise and routing must be unaffected.
        let subs = ConversationSubscriptions::new();
        let viewer = Arc::new(RecordingSink::default());
        subs.register("viewer", "alice", viewer.clone());
        subs.set_subscriptions("viewer", vec!["c1".into()]);

        subs.route("c1", &delta("c1"), "origin", "alice").await;

        assert_eq!(
            viewer.0.lock().unwrap().len(),
            1,
            "routing must work with no observer installed"
        );
        subs.unregister("viewer"); // must not panic without an observer
    }
}
