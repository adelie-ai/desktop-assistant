//! Daemon adapter for the parent-wake feature (#668).
//!
//! [`ParentWakeCoordinator`] (in `application`) owns *when* to wake a parent
//! conversation; this module owns *how*. [`HandlerParentWaker`] implements the
//! coordinator's [`ParentWaker`] port by injecting the wake prompt as a fresh
//! turn on the parent conversation through the handler's normal
//! `handle_send_message_with_override_for` path — the same path a client send
//! takes — so the turn registers as a `TaskKind::Conversation`, streams, and
//! (via the handler's [`FanOutSink`]) renders live in every client viewing that
//! conversation, even though no client originated it.
//!
//! [`ParentWaker`]: desktop_assistant_application::parent_wake::ParentWaker
//! [`ParentWakeCoordinator`]: desktop_assistant_application::parent_wake::ParentWakeCoordinator
//! [`FanOutSink`]: desktop_assistant_application::conversation_subs::FanOutSink

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use desktop_assistant_application::parent_wake::{NullEventSink, ParentWaker};
use desktop_assistant_application::{AssistantApiHandler, EventSink, RequestContext};
use desktop_assistant_auth_jwt::UserId;

/// Drives a wake turn over the handler's send path. Holds a `Weak` to the
/// handler because the handler (transitively, via the registry's observer)
/// owns the coordinator that owns this waker — a strong ref would leak the whole
/// graph for the process lifetime.
pub struct HandlerParentWaker {
    handler: Weak<dyn AssistantApiHandler>,
}

impl HandlerParentWaker {
    /// Wrap a `Weak` downgrade of the daemon's shared api handler.
    pub fn new(handler: Weak<dyn AssistantApiHandler>) -> Self {
        Self { handler }
    }
}

#[async_trait]
impl ParentWaker for HandlerParentWaker {
    async fn wake(&self, user_id: UserId, conversation_id: String, prompt: String) {
        let Some(handler) = self.handler.upgrade() else {
            // Handler dropped (shutdown). Nothing to wake.
            tracing::debug!("parent-wake: api handler gone; skipping wake turn");
            return;
        };
        let ctx = RequestContext::for_user(user_id);
        // Synthetic, unique request id: no client is correlating this turn, but
        // the send path stamps every event with it and it must not collide with
        // a real client request.
        let request_id = format!("parent-wake-{}", uuid::Uuid::new_v4());
        let sink: Arc<dyn EventSink> = Arc::new(NullEventSink);
        if let Err(e) = handler
            .handle_send_message_with_override_for(
                ctx,
                conversation_id,
                prompt,
                None,          // no model override — use the conversation's default
                String::new(), // no per-turn system refinement
                request_id,
                sink,
            )
            .await
        {
            // A failed wake turn is logged and swallowed: it must never poison
            // the coordinator or abort the daemon. The parent simply isn't
            // re-engaged this time; the results remain on the scratchpad.
            tracing::warn!(error = %e, "parent-wake turn failed");
        }
    }
}
