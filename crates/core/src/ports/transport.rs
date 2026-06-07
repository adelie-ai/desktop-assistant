//! Request-scoped transport context (issue #243).
//!
//! Tool execution-locality routing needs to know whether the connection
//! driving the current turn is co-located with the daemon. A local transport
//! (Unix-domain socket or D-Bus) can only be reached from the daemon's own
//! machine, so a client-registered tool on such a connection runs on the same
//! host as the server-side MCP tools — the [`crate::domain::ToolLocality`]
//! distinction collapses to "this machine". A WebSocket connection may be
//! remote, so the two localities stay distinct.
//!
//! This module exposes a task-local [`crate::domain::TransportKind`] slot,
//! mirroring [`crate::ports::auth`]'s `UserId` plumbing: the transport adapter
//! installs the connection's kind via [`with_transport_kind`] before invoking
//! the handler, and the dispatch loop reads it via [`current_transport_kind`]
//! when it assembles the per-turn tool set.
//!
//! ## System-id co-location (#248)
//!
//! Phase 2 refines co-location with an exact per-machine **system id**: the
//! daemon compares the client's reported id to its own and installs the
//! authoritative result via [`with_co_location`] alongside the transport. The
//! dispatch loop reads it via [`current_co_location`]; when the client reported
//! no id (an older client) the slot holds `None` and co-location falls back to
//! the transport heuristic, preserving Phase-1 behaviour. A client-supplied
//! host label (for a friendlier tool note) rides the same path via
//! [`with_client_label`] / [`current_client_label`].
//!
//! ## Default
//!
//! When no scope is installed — tests, dreaming jobs, and any caller that does
//! not route through a transport adapter — [`current_transport_kind`] returns
//! [`TransportKind::Uds`]. UDS is the live default transport and is co-located,
//! so the safe, common-case behaviour (treat tools as same-machine) applies
//! without every test having to install a scope. [`current_co_location`]
//! defaults to `None` (no authoritative result ⇒ fall back to transport) and
//! [`current_client_label`] to `None`.

use crate::domain::TransportKind;

tokio::task_local! {
    /// The transport the current turn's connection arrived on. Installed by
    /// the transport adapter via [`with_transport_kind`]; read by the dispatch
    /// loop via [`current_transport_kind`]. Unset outside a transport scope,
    /// which [`current_transport_kind`] reports as [`TransportKind::Uds`] (the
    /// co-located default).
    static TRANSPORT_KIND: TransportKind;

    /// The authoritative per-machine system-id co-location result for the
    /// current connection (#248): `Some(true)`/`Some(false)` when the client
    /// reported an id the daemon could compare to its own, `None` for an older
    /// client that sent none (⇒ fall back to the transport heuristic).
    /// Installed by the transport adapter via [`with_co_location`]; read via
    /// [`current_co_location`].
    static CO_LOCATION: Option<bool>;

    /// A client-supplied host label for the current connection (#248), used to
    /// make the remote tool note friendlier (e.g. `your device 'laptop'`).
    /// `None` when the client sent none. Installed via [`with_client_label`];
    /// read via [`current_client_label`].
    static CLIENT_LABEL: Option<String>;
}

/// Run `fut` with `kind` installed as the current task-local transport. All
/// [`current_transport_kind`] calls inside the future (and any sub-tasks that
/// inherit the scope) observe `kind`.
///
/// Note: like every `tokio::task_local!`, the slot does **not** cross a
/// `tokio::spawn` boundary. Adapters whose turn body runs on a spawned task
/// must thread the value explicitly and re-install it inside the spawn (the
/// same discipline `with_user_id` follows).
pub async fn with_transport_kind<F, T>(kind: TransportKind, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TRANSPORT_KIND.scope(kind, fut).await
}

/// The current task-local transport kind, or [`TransportKind::Uds`] when no
/// scope is installed. Safe to call from any async context — never panics,
/// never blocks. The UDS default means callers that don't route through a
/// transport adapter treat tools as co-located, which is the live common case.
pub fn current_transport_kind() -> TransportKind {
    TRANSPORT_KIND
        .try_with(|k| *k)
        .unwrap_or(TransportKind::Uds)
}

/// Run `fut` with `co_located` installed as the current authoritative
/// system-id co-location result (#248). `Some(true)`/`Some(false)` overrides
/// the transport heuristic; `None` defers to it. Like every task-local it does
/// not cross `tokio::spawn` — re-install inside spawned turn bodies.
pub async fn with_co_location<F, T>(co_located: Option<bool>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CO_LOCATION.scope(co_located, fut).await
}

/// The current authoritative system-id co-location result (#248), or `None`
/// when no scope is installed (⇒ the caller falls back to the transport
/// heuristic). Never panics or blocks.
pub fn current_co_location() -> Option<bool> {
    CO_LOCATION.try_with(|c| *c).unwrap_or(None)
}

/// Run `fut` with `label` installed as the current client-supplied host label
/// (#248). `None` means the client sent none. Like every task-local it does not
/// cross `tokio::spawn` — re-install inside spawned turn bodies.
pub async fn with_client_label<F, T>(label: Option<String>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CLIENT_LABEL.scope(label, fut).await
}

/// The current client-supplied host label (#248), or `None` when no scope is
/// installed or the client sent none. Never panics or blocks.
pub fn current_client_label() -> Option<String> {
    CLIENT_LABEL.try_with(|l| l.clone()).unwrap_or(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn current_transport_kind_defaults_to_uds_outside_scope() {
        assert_eq!(current_transport_kind(), TransportKind::Uds);
    }

    #[tokio::test]
    async fn current_transport_kind_observes_installed_scope() {
        let observed =
            with_transport_kind(TransportKind::WebSocket, async { current_transport_kind() }).await;
        assert_eq!(observed, TransportKind::WebSocket);
        // After the scope exits the slot is unset again (back to the default).
        assert_eq!(current_transport_kind(), TransportKind::Uds);
    }

    #[tokio::test]
    async fn nested_transport_kind_shadows_outer() {
        let observed = with_transport_kind(TransportKind::Uds, async {
            with_transport_kind(TransportKind::WebSocket, async { current_transport_kind() }).await
        })
        .await;
        assert_eq!(observed, TransportKind::WebSocket);
    }

    #[tokio::test]
    async fn spawned_task_outside_scope_sees_default() {
        // task_local slots don't cross `tokio::spawn`.
        let observed = tokio::spawn(async { current_transport_kind() })
            .await
            .unwrap();
        assert_eq!(observed, TransportKind::Uds);
    }

    #[tokio::test]
    async fn current_co_location_defaults_to_none_outside_scope() {
        assert_eq!(current_co_location(), None);
    }

    #[tokio::test]
    async fn current_co_location_observes_installed_scope() {
        let observed = with_co_location(Some(true), async { current_co_location() }).await;
        assert_eq!(observed, Some(true));
        let observed = with_co_location(Some(false), async { current_co_location() }).await;
        assert_eq!(observed, Some(false));
        // After the scope exits the slot is unset again (back to the default).
        assert_eq!(current_co_location(), None);
    }

    #[tokio::test]
    async fn current_client_label_defaults_to_none_and_observes_scope() {
        assert_eq!(current_client_label(), None);
        let observed =
            with_client_label(Some("laptop".to_string()), async { current_client_label() }).await;
        assert_eq!(observed.as_deref(), Some("laptop"));
        assert_eq!(current_client_label(), None);
    }
}
