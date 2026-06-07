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
//! ## Default
//!
//! When no scope is installed — tests, dreaming jobs, and any caller that does
//! not route through a transport adapter — [`current_transport_kind`] returns
//! [`TransportKind::Uds`]. UDS is the live default transport and is co-located,
//! so the safe, common-case behaviour (treat tools as same-machine) applies
//! without every test having to install a scope.

use crate::domain::TransportKind;

tokio::task_local! {
    /// The transport the current turn's connection arrived on. Installed by
    /// the transport adapter via [`with_transport_kind`]; read by the dispatch
    /// loop via [`current_transport_kind`]. Unset outside a transport scope,
    /// which [`current_transport_kind`] reports as [`TransportKind::Uds`] (the
    /// co-located default).
    static TRANSPORT_KIND: TransportKind;
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
}
