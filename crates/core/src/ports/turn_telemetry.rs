//! What a turn calls itself in telemetry, and where it dispatched.
//!
//! Two request-scoped values that the turn loop reads but does not own.
//!
//! ## The correlation id
//!
//! The transport dispatcher mints one uuid per turn and stamps every streamed
//! event with it, so a client already correlates its own response stream by
//! that value. [`with_request_id`] puts the same value where the turn loop can
//! read it, and the loop puts it on the turn span. Every log line written
//! inside that span then carries it through the console layer's span scope,
//! with no threading by hand.
//!
//! Without the slot installed - a background job, an agent run, a test -
//! [`current_request_id`] answers `None` and the turn span records the
//! [`UNSET`] sentinel. A turn is still traceable by its conversation id in
//! that case; it simply has no client-side identifier to be pasted from.
//!
//! ## The route
//!
//! [`TurnRoute`] is which connection, which connector and which model the turn
//! dispatches on. The daemon resolves all three once per turn, before it calls
//! into the core loop, and the loop reads them for the turn span, for the
//! per-round token metrics and for the completion line.
//!
//! They arrive together because they are resolved together and because a
//! metric labelled with one of them and not the others cannot be read. A
//! provider name without a model name does not say which model was slow.
//!
//! ## Why task-locals
//!
//! The same reason as [`crate::ports::auth`] and [`crate::ports::session`]:
//! the values are request-scoped, cross many `await` points and several
//! layers, and are read by code that would otherwise need a new parameter on
//! every port method between the transport and the loop.
//!
//! Like every `tokio::task_local!`, neither slot crosses a `tokio::spawn`.
//! Both are installed *inside* the spawned turn body - the request id by the
//! application layer's turn runner, the route by the daemon's routing handler -
//! so neither needs to ride [`crate::ports::request_scope::RequestScope`].

use std::future::Future;

/// What a span field or a metric label reads when a value was not resolved.
///
/// A named sentinel rather than an empty string: an empty field renders as
/// nothing at all, which reads on a console line as though the field were
/// missing rather than unresolved. It is also `&'static str`, so it can never
/// widen a metric's label cardinality.
pub const UNSET: &str = "unset";

tokio::task_local! {
    /// The turn's client-facing correlation id. See the module header.
    static REQUEST_ID: String;
}

tokio::task_local! {
    /// Where the turn dispatches. See the module header.
    static TURN_ROUTE: TurnRoute;
}

/// Which connection, connector and model a turn dispatches on.
///
/// Every field is optional because the daemon's routing has a documented
/// fall-through: when no concrete live connection resolves, the turn goes to
/// the statically-configured primary client and the daemon does not know which
/// connection or model that is. `None` is that state, and it is reported as
/// [`UNSET`] rather than guessed at.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnRoute {
    /// The configured connection the turn dispatches through.
    pub connection_id: Option<String>,
    /// The connector kind behind that connection - `anthropic`, `ollama` and
    /// so on. This is the `provider` axis of the token and latency metrics.
    pub provider: Option<String>,
    /// The model id the turn pins for this dispatch.
    pub model: Option<String>,
}

impl TurnRoute {
    /// The connection id, or [`UNSET`].
    pub fn connection_id(&self) -> &str {
        self.connection_id.as_deref().unwrap_or(UNSET)
    }

    /// The connector kind, or [`UNSET`].
    pub fn provider(&self) -> &str {
        self.provider.as_deref().unwrap_or(UNSET)
    }

    /// The model id, or [`UNSET`].
    pub fn model(&self) -> &str {
        self.model.as_deref().unwrap_or(UNSET)
    }
}

/// Run `fut` with `request_id` installed as the turn's correlation id.
pub async fn with_request_id<F, T>(request_id: String, fut: F) -> T
where
    F: Future<Output = T>,
{
    REQUEST_ID.scope(request_id, fut).await
}

/// The turn's correlation id, or `None` outside any turn scope.
pub fn current_request_id() -> Option<String> {
    REQUEST_ID.try_with(Clone::clone).ok()
}

/// Run `fut` with `route` installed as the turn's dispatch route.
pub async fn with_turn_route<F, T>(route: TurnRoute, fut: F) -> T
where
    F: Future<Output = T>,
{
    TURN_ROUTE.scope(route, fut).await
}

/// The turn's dispatch route. An all-`None` route outside any turn scope,
/// which reports as [`UNSET`] on every axis rather than as a guess.
pub fn current_turn_route() -> TurnRoute {
    TURN_ROUTE.try_with(Clone::clone).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_unscoped_context_has_no_request_id() {
        assert_eq!(current_request_id(), None);
    }

    #[tokio::test]
    async fn the_request_id_is_readable_inside_its_scope() {
        let observed = with_request_id("req-1".to_string(), async { current_request_id() }).await;
        assert_eq!(observed, Some("req-1".to_string()));
        assert_eq!(
            current_request_id(),
            None,
            "the slot must not leak past its scope"
        );
    }

    #[tokio::test]
    async fn an_unresolved_route_reports_unset_on_every_axis() {
        let route = current_turn_route();
        assert_eq!(route.connection_id(), UNSET);
        assert_eq!(route.provider(), UNSET);
        assert_eq!(route.model(), UNSET);
    }

    #[tokio::test]
    async fn the_route_is_readable_inside_its_scope() {
        let installed = TurnRoute {
            connection_id: Some("conn-a".to_string()),
            provider: Some("anthropic".to_string()),
            model: Some("claude-example".to_string()),
        };
        let observed = with_turn_route(installed.clone(), async { current_turn_route() }).await;
        assert_eq!(observed, installed);
        assert_eq!(
            current_turn_route(),
            TurnRoute::default(),
            "the slot must not leak past its scope"
        );
    }
}
