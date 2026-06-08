//! Request-scoped login-session identity (#261).
//!
//! A *session* is a single client connection (one UDS/WebSocket login).
//! Unlike [`crate::ports::auth::UserId`] — which is shared by every
//! connection a user opens — a [`SessionId`] is unique per connection, so
//! state that must NOT bleed between two windows of the same user keys on
//! it. The motivating case is client-local tool registration (#107/#234):
//! the voice daemon registers `say_this`; without a per-session key that
//! tool was offered on a *text* client's turns (same user) and wedged the
//! conversation when the text client couldn't fulfil it (#260). Two TUIs
//! logged in as one user must have independent tool sets — exactly what a
//! per-connection id provides and a per-user id cannot.
//!
//! ## Why a task-local
//!
//! Same rationale as [`crate::ports::auth`]: the value is request-scoped,
//! crosses many `await` points and layers, and is read by code (the
//! client-tool coordinator) that doesn't otherwise know about the
//! transport pipeline. Threading a `session_id` through every port method
//! would touch surfaces that don't otherwise care. The transport
//! dispatcher installs it via [`with_session_id`] once per request,
//! alongside [`crate::ports::auth::with_user_id`].
//!
//! ## Unscoped fallback
//!
//! When nothing installed the slot — background workers, tests, and the
//! D-Bus path (which cannot register client tools at all) —
//! [`current_session_id`] returns [`SessionId::unscoped`]. Registration
//! only ever happens on a socket transport that installs a real id, so
//! the unscoped bucket is never *written* by a real client; readers in
//! unscoped contexts simply observe an empty set.

/// Identity of a single client connection (login session).
///
/// Opaque and process-local: minted per accepted connection by the
/// transport layer, never persisted, never sent on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl SessionId {
    /// Wrap a transport-minted session identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The sentinel for an unscoped context (no connection installed).
    /// Distinct from any real connection id so the unscoped bucket can
    /// never collide with a live session.
    pub fn unscoped() -> Self {
        Self("unscoped".to_string())
    }

    /// Borrow the underlying identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::unscoped()
    }
}

tokio::task_local! {
    /// Per-connection session identity. Installed by the transport
    /// dispatcher via [`with_session_id`] for every request on the
    /// connection; read by the client-tool coordinator via
    /// [`current_session_id`] so each connection's registered tools are
    /// visible only to its own turns.
    static SESSION_ID: SessionId;
}

/// Run `fut` with `session_id` installed as the current task-local
/// session identity. All [`current_session_id`] calls inside the future
/// observe `session_id`. The dispatcher re-installs it on spawned
/// send-message tasks because `task_local`s don't cross `tokio::spawn`.
pub async fn with_session_id<F, T>(session_id: SessionId, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    SESSION_ID.scope(session_id, fut).await
}

/// The current task-local session identity, or [`SessionId::unscoped`]
/// when no connection scope is installed. Never panics, never blocks.
pub fn current_session_id() -> SessionId {
    SESSION_ID.try_with(|s| s.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn current_session_id_outside_scope_is_unscoped() {
        assert_eq!(current_session_id(), SessionId::unscoped());
        assert_eq!(current_session_id().as_str(), "unscoped");
    }

    #[tokio::test]
    async fn current_session_id_inside_scope_returns_installed_value() {
        let observed =
            with_session_id(SessionId::new("conn-7"), async { current_session_id() }).await;
        assert_eq!(observed, SessionId::new("conn-7"));
    }

    #[tokio::test]
    async fn nested_scopes_override_then_restore() {
        let (inner, after) = with_session_id(SessionId::new("outer"), async {
            let inner =
                with_session_id(SessionId::new("inner"), async { current_session_id() }).await;
            (inner, current_session_id())
        })
        .await;
        assert_eq!(inner, SessionId::new("inner"));
        assert_eq!(after, SessionId::new("outer"));
    }

    #[tokio::test]
    async fn spawned_task_outside_scope_falls_back_to_unscoped() {
        // `task_local`s don't cross `tokio::spawn`; the dispatcher
        // re-installs on spawned send tasks for exactly this reason.
        let observed = tokio::spawn(async { current_session_id() }).await.unwrap();
        assert_eq!(observed, SessionId::unscoped());
    }
}
