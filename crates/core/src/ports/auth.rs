//! Request-scoped auth context (#105).
//!
//! This module exposes a task-local [`UserId`] slot so storage layers
//! can scope SQL to the requesting user without every port-trait method
//! growing a `user_id: &UserId` parameter. The handler at the transport
//! boundary (WS, UDS, future API Gateway WS) extracts the JWT `sub`,
//! wraps it in [`UserId`], and installs it via [`with_user_id`] before
//! invoking any [`crate::ports::inbound`] service. The storage adapters
//! read it via [`current_user_id`] at SQL composition time.
//!
//! ## Why a task-local
//!
//! The codebase already threads per-turn LLM configuration the same way
//! (see [`crate::ports::llm::CONTEXT_BUDGET`] and [`MODEL_OVERRIDE`]).
//! User identity is the same shape of value: scoped to a single request,
//! crossing many `await` points and many layers, read by code that
//! doesn't otherwise know about the auth pipeline. Threading it through
//! every signature would touch every port trait and every test fixture
//! for a value that doesn't influence those signatures' shape — task-
//! local keeps the existing API surface stable while still making the
//! identity available where storage needs it.
//!
//! Independently shippable single-tenant mode: when nothing has set the
//! task-local, [`current_user_id`] returns [`UserId::default`], which
//! resolves to the schema-level sentinel `"default"`. That's the same
//! value the migration in #102 writes as the column default and as the
//! backfill for pre-multi-tenant rows. The result is that single-tenant
//! desktop deploys see no behavioural change.

pub use desktop_assistant_auth_jwt::{DEFAULT_USER_ID, UserId};

tokio::task_local! {
    /// Per-request user identity. Set by the transport-adapter handler
    /// via [`with_user_id`] right after JWT validation; read by storage
    /// adapters via [`current_user_id`] when composing SQL.
    ///
    /// When this slot is unset (background workers, tests that don't go
    /// through a transport, single-tenant deploys without JWT auth),
    /// [`current_user_id`] returns [`UserId::default`] which maps to the
    /// schema sentinel `"default"`. This keeps every code path
    /// dual-mode: a single-tenant install collapses to the sentinel; a
    /// multi-tenant install sees the JWT `sub` as the user id; the SQL
    /// is identical in both cases.
    static USER_ID: UserId;
}

/// Run `fut` with `user_id` installed as the current task-local user
/// identity. All [`current_user_id`] calls inside the future (and any
/// sub-tasks that inherit the scope) observe `user_id`.
///
/// The transport handler ([`crates/ws-interface`] and friends) calls
/// this exactly once per request, immediately after extracting the
/// identity from the JWT. Nested calls override the outer scope for the
/// duration of the inner future — this is how dreaming workers and
/// embedding backfill jobs that iterate over many users can scope each
/// per-user pass.
pub async fn with_user_id<F, T>(user_id: UserId, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    USER_ID.scope(user_id, fut).await
}

/// The current task-local user identity, or [`UserId::default`] (the
/// schema sentinel) when no scope is installed.
///
/// Storage adapters call this at SQL composition time. Safe to call
/// from any async context — never panics, never blocks.
pub fn current_user_id() -> UserId {
    USER_ID.try_with(|u| u.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn current_user_id_outside_scope_is_default() {
        // Single-tenant / unscoped path: storage code that runs without
        // a transport-installed scope falls through to the sentinel.
        assert_eq!(current_user_id(), UserId::default());
        assert_eq!(current_user_id().as_str(), "default");
    }

    #[tokio::test]
    async fn current_user_id_inside_scope_returns_installed_value() {
        let observed = with_user_id(UserId::new("alice"), async {
            current_user_id()
        })
        .await;
        assert_eq!(observed, UserId::new("alice"));
    }

    #[tokio::test]
    async fn nested_scopes_override_then_restore() {
        // Inner scope wins for the duration of the inner future; outer
        // scope restores afterwards. Mirrors how a per-user dreaming
        // worker would loop inside a daemon-wide background task.
        let result = with_user_id(UserId::new("outer"), async {
            let inner = with_user_id(UserId::new("inner"), async {
                current_user_id()
            })
            .await;
            let after = current_user_id();
            (inner, after)
        })
        .await;
        assert_eq!(result.0, UserId::new("inner"));
        assert_eq!(result.1, UserId::new("outer"));
    }

    #[tokio::test]
    async fn current_user_id_outside_any_scope_after_spawn_falls_back_to_default() {
        // Tasks spawned outside any `with_user_id` scope don't see the
        // slot — that's the normal `task_local` semantic, and it's the
        // reason the dreaming worker re-installs the scope around each
        // per-user batch. Documenting the contract here keeps callers
        // honest: anything that crosses `tokio::spawn` must
        // re-install before touching storage.
        let observed = tokio::spawn(async { current_user_id() })
            .await
            .unwrap();
        assert_eq!(
            observed,
            UserId::default(),
            "spawned tasks outside any scope must fall through to the sentinel"
        );
    }
}
