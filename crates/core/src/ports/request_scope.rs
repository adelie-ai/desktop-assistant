//! One bundle for the request-scoped task-locals that must cross a
//! `tokio::spawn` boundary (issue #305 item 4).
//!
//! ## The problem this solves
//!
//! A handful of per-request values propagate through the turn as
//! `tokio::task_local!`s — the user id (#105), the login-session id (#261),
//! the connection's transport kind plus its system-id co-location result and
//! client host label (#243/#248). The transport dispatcher installs all of
//! them around each request. But `task_local`s do **not** cross a
//! `tokio::spawn`, and the streaming send-message path spawns the turn body on
//! a fresh task. So every spawn site has to *capture each value before the
//! spawn and re-install it inside the spawned body*.
//!
//! Doing that by hand — one `with_*` wrap per local, per spawn site — is the
//! bug that issue #261 was: a new task-local (the session id) was added and the
//! re-install was simply forgotten at one spawn site, so that path silently saw
//! the unscoped default. Co-location and the client label were dropped across
//! the spawn the same way. The failure is invisible (a *default* value, not a
//! panic) and only shows up as subtly wrong behaviour far downstream.
//!
//! ## The fix
//!
//! [`RequestScope`] is a single struct holding all of those values, with:
//!
//! - [`RequestScope::capture`] — read every spawn-crossing task-local into the
//!   bundle (call this *before* the spawn, while the dispatcher's scopes are
//!   still installed), and
//! - [`RequestScope::scope`] — re-install all of them around a future in one
//!   call (call this *inside* the spawned body).
//!
//! Now there is exactly **one** place to add a new spawn-crossing local: a
//! field on this struct, captured and re-installed in lock-step. A site that
//! captures-then-scopes can never drop a value, so the missed-re-install class
//! of bug (#261) becomes a compile-time non-issue.
//!
//! This is purely a *consolidation*: the individual `with_*`/`current_*`
//! accessors are unchanged and still used directly by the dispatcher (which
//! installs the locals from its `AuthContext`, not from a captured scope) and
//! by tests. `RequestScope` is the spawn-crossing convenience, not a
//! replacement for the per-local API.

use std::future::Future;

use crate::domain::TransportKind;
use crate::ports::auth::{UserId, current_user_id, with_user_id};
use crate::ports::session::{SessionId, current_session_id, with_session_id};
use crate::ports::transport::{
    current_client_label, current_co_location, current_transport_kind, with_client_label,
    with_co_location, with_transport_kind,
};

/// The set of request-scoped task-locals that must be re-installed inside a
/// spawned turn body. Capture it before the spawn, re-install it inside.
///
/// Every field corresponds to one `tokio::task_local!` that the transport
/// dispatcher installs around a request but that would otherwise be lost across
/// the `tokio::spawn` that runs the streaming turn body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestScope {
    /// Per-request user identity (#105). Defaults to [`UserId::default`] (the
    /// `"default"` schema sentinel) outside any scope.
    pub user_id: UserId,
    /// Per-connection login-session identity (#261). Defaults to
    /// [`SessionId::unscoped`] outside any scope.
    pub session_id: SessionId,
    /// The transport the connection arrived on (#243). Defaults to
    /// [`TransportKind::Uds`] (co-located) outside any scope.
    pub transport: TransportKind,
    /// Authoritative system-id co-location result (#248): `Some(true)` /
    /// `Some(false)` when the client reported a comparable id, `None` to defer
    /// to the transport heuristic.
    pub co_located: Option<bool>,
    /// Client-reported host label (#248) for a friendlier remote tool note;
    /// `None` when the client sent none.
    pub client_label: Option<String>,
}

impl RequestScope {
    /// Snapshot the current request-scoped task-locals into a bundle.
    ///
    /// Call this *before* a `tokio::spawn` while the dispatcher's `with_*`
    /// scopes are still installed on the current task. Each field falls back to
    /// the same default the individual `current_*` accessor uses when its slot
    /// is unset, so capturing outside any scope yields a coherent
    /// all-defaults bundle (matching the unscoped single-tenant path).
    pub fn capture() -> Self {
        Self {
            user_id: current_user_id(),
            session_id: current_session_id(),
            transport: current_transport_kind(),
            co_located: current_co_location(),
            client_label: current_client_label(),
        }
    }

    /// Run `fut` with every captured task-local re-installed.
    ///
    /// Call this *inside* a spawned turn body so the turn sees the same
    /// request scope its dispatcher installed. The nesting order matches the
    /// dispatcher's (and is immaterial — the locals are independent slots).
    ///
    /// `current_user_id()`, `current_session_id()`, `current_transport_kind()`,
    /// `current_co_location()`, and `current_client_label()` inside `fut` all
    /// observe the captured values.
    pub async fn scope<F, T>(self, fut: F) -> T
    where
        F: Future<Output = T>,
    {
        let Self {
            user_id,
            session_id,
            transport,
            co_located,
            client_label,
        } = self;
        with_co_location(
            co_located,
            with_client_label(
                client_label,
                with_transport_kind(
                    transport,
                    with_user_id(user_id, with_session_id(session_id, fut)),
                ),
            ),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn capture_outside_any_scope_is_all_defaults() {
        let scope = RequestScope::capture();
        assert_eq!(scope.user_id, UserId::default());
        assert_eq!(scope.session_id, SessionId::unscoped());
        assert_eq!(scope.transport, TransportKind::Uds);
        assert_eq!(scope.co_located, None);
        assert_eq!(scope.client_label, None);
    }

    #[tokio::test]
    async fn capture_then_scope_round_trips_every_local() {
        // Install all five locals, capture them into a bundle, then (mimicking
        // a spawn boundary that drops them) re-install from the bundle and
        // confirm every value is observed — including co_location and the
        // client label, the two that the hand-written spawn sites dropped.
        let captured = with_co_location(
            Some(true),
            with_client_label(
                Some("laptop".to_string()),
                with_transport_kind(
                    TransportKind::WebSocket,
                    with_user_id(
                        UserId::new("alice"),
                        with_session_id(SessionId::new("sess-9"), async {
                            RequestScope::capture()
                        }),
                    ),
                ),
            ),
        )
        .await;

        assert_eq!(captured.user_id, UserId::new("alice"));
        assert_eq!(captured.session_id, SessionId::new("sess-9"));
        assert_eq!(captured.transport, TransportKind::WebSocket);
        assert_eq!(captured.co_located, Some(true));
        assert_eq!(captured.client_label, Some("laptop".to_string()));

        // Re-install from the captured bundle in a context where none of the
        // locals are set (simulating the post-spawn task) and read them back.
        let observed = captured
            .clone()
            .scope(async {
                (
                    current_user_id(),
                    current_session_id(),
                    current_transport_kind(),
                    current_co_location(),
                    current_client_label(),
                )
            })
            .await;

        assert_eq!(observed.0, UserId::new("alice"));
        assert_eq!(observed.1, SessionId::new("sess-9"));
        assert_eq!(observed.2, TransportKind::WebSocket);
        assert_eq!(observed.3, Some(true));
        assert_eq!(observed.4, Some("laptop".to_string()));
    }

    #[tokio::test]
    async fn scope_does_not_leak_past_the_future() {
        let scope = RequestScope {
            user_id: UserId::new("bob"),
            session_id: SessionId::new("s1"),
            transport: TransportKind::WebSocket,
            co_located: Some(false),
            client_label: Some("phone".to_string()),
        };
        scope.scope(async {}).await;

        // After the scoped future returns, the ambient task sees defaults again.
        assert_eq!(current_user_id(), UserId::default());
        assert_eq!(current_session_id(), SessionId::unscoped());
        assert_eq!(current_transport_kind(), TransportKind::Uds);
        assert_eq!(current_co_location(), None);
        assert_eq!(current_client_label(), None);
    }
}
