//! What a turn calls itself in telemetry, where it dispatched, and which trace
//! it belongs to.
//!
//! Three request-scoped values that the turn loop reads but does not own.
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
//! ## The trace
//!
//! [`TurnTrace`] is the trace the turn belongs to, and the conversation it is
//! part of. A turn is one trace: the root span opens when a person commits an
//! input and closes when the reply is complete. A conversation is **not** a
//! trace - it lives for days and holds an unbounded number of turns, which no
//! backend renders usefully - so the conversation id rides here as an attribute
//! that every span in the turn carries.
//!
//! The trace id is the turn's own correlation id, reinterpreted: a uuid is 16
//! bytes and a W3C trace id is 16 bytes, so
//! [`adelie_telemetry::trace_id_from_uuid`] makes them the same value with no
//! mapping table. One identifier then appears in the client's event stream, in
//! the daemon's log and in a trace backend. That holds with the `otel` feature
//! off, so a default build stays correlatable without exporting anything.
//!
//! Who resolves it: the transport boundary, which knows both what the client
//! sent and what the daemon adopted. [`adopt_or_mint_turn_id`] is the rule for
//! the correlation id and [`resolve_turn_trace`] is the rule for the trace.
//! A turn that reaches the loop by another door - an agent run, a scheduled
//! job, a test - resolves its own from its request id, so no turn is ever
//! without a trace.
//!
//! ## Why task-locals
//!
//! The same reason as [`crate::ports::auth`] and [`crate::ports::session`]:
//! the values are request-scoped, cross many `await` points and several
//! layers, and are read by code that would otherwise need a new parameter on
//! every port method between the transport and the loop.
//!
//! Like every `tokio::task_local!`, no slot crosses a `tokio::spawn`. The
//! request id and the route are installed *inside* the spawned turn body - the
//! request id by the application layer's turn runner, the route by the daemon's
//! routing handler - so neither needs to ride
//! [`crate::ports::request_scope::RequestScope`]. The trace is different: it is
//! resolved at the transport boundary, before the spawn, so it rides that
//! bundle like every other spawn-crossing value.

use std::future::Future;

use adelie_telemetry::trace_context::{TraceOrigin, TraceParent, resolve_trace_or_mint};

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

tokio::task_local! {
    /// The trace the turn belongs to. See the module header.
    static TURN_TRACE: Option<TurnTrace>;
}

/// Which trace a turn belongs to, and which conversation it is part of.
///
/// Both values are on every span the turn opens: the trace id because that is
/// what joins this process's spans to the client's and to each MCP server's,
/// and the conversation id because a conversation is an attribute rather than
/// a trace of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnTrace {
    /// The trace, and whether it was continued from a caller or minted here.
    pub trace: TraceOrigin,
    /// The conversation this turn is part of.
    pub conversation_id: String,
}

impl TurnTrace {
    /// The trace for a turn that starts here, derived from its correlation id.
    ///
    /// Used where no caller supplied a trace: an agent run, a scheduled job, a
    /// test. `request_id` that is not a usable uuid produces a generated trace
    /// id rather than no trace at all.
    pub fn minted(request_id: Option<&str>, conversation_id: &str) -> Self {
        Self {
            trace: resolve_trace_or_mint(None, uuid_bytes(request_id)),
            conversation_id: conversation_id.to_string(),
        }
    }

    /// The `traceparent` header value naming this turn's root.
    ///
    /// Deterministic, and derived from the trace id alone, so a process with no
    /// span machinery of its own still produces a header a receiver can join.
    pub fn root_header(&self) -> String {
        TraceParent::root_for(self.trace.trace_id(), true).to_header()
    }
}

/// The turn's correlation id: the client's own, when it sent a usable one.
///
/// A turn starts when a person commits an input, so the client is the top of
/// it and mints the id. The daemon adopts that id and mints its own only when
/// none arrives, which is what keeps an older client working unchanged.
///
/// `supplied` is accepted when it parses as a uuid and is not the nil uuid,
/// which is the same rule [`adelie_telemetry::trace_id_from_uuid`] applies. The
/// answer is the canonical hyphenated lowercase rendering, so one turn has one
/// spelling wherever it is read; the daemon returns it on the ack, so a client
/// that spelled it differently still learns the value its events carry.
/// Anything else falls back to minting rather than failing the turn.
///
/// This value is a correlation id and nothing else. It grants no capability and
/// names no user, so it must reach no authorization or tenancy decision - the
/// next reader will wonder, and the answer is that a client choosing its own
/// value could otherwise choose its own permissions. It is also not the
/// idempotency key: that is a separate field on the same command, and the
/// exactly-once retry path is what reads it.
pub fn adopt_or_mint_turn_id(supplied: Option<&str>) -> String {
    match supplied.and_then(parse_turn_id) {
        Some(id) => id,
        None => uuid::Uuid::new_v4().to_string(),
    }
}

/// The trace a turn belongs to, resolved at the transport boundary.
///
/// An incoming `traceparent` wins, because a caller that already has a trace
/// should be continued rather than restarted. Otherwise the turn's own
/// correlation id becomes the trace id. Neither input can fail the turn: a
/// malformed header is discarded with a WARN naming the reason, and an unusable
/// request id produces a generated trace id.
pub fn resolve_turn_trace(
    incoming_traceparent: Option<&str>,
    request_id: &str,
    conversation_id: &str,
) -> TurnTrace {
    TurnTrace {
        trace: resolve_trace_or_mint(incoming_traceparent, uuid_bytes(Some(request_id))),
        conversation_id: conversation_id.to_string(),
    }
}

/// Run `fut` with `trace` installed as the turn's trace.
pub async fn with_turn_trace<F, T>(trace: Option<TurnTrace>, fut: F) -> T
where
    F: Future<Output = T>,
{
    TURN_TRACE.scope(trace, fut).await
}

/// The turn's trace, or `None` outside any turn scope.
pub fn current_turn_trace() -> Option<TurnTrace> {
    TURN_TRACE.try_with(Clone::clone).ok().flatten()
}

/// The conversation every span in this turn carries, or [`UNSET`].
pub fn current_conversation_id() -> String {
    current_turn_trace().map_or_else(|| UNSET.to_string(), |trace| trace.conversation_id)
}

/// Make `span` carry this turn's trace id.
///
/// A host that opens its own root span for a turn - a desktop client, the web
/// BFF - calls this so its span and the daemon's land in one trace rather than
/// two that share a timestamp. The daemon's turn loop calls it for the same
/// reason.
///
/// With the `otel` feature off a span has no trace id to carry and this does
/// nothing. Correlation still works in that build, because the trace id is the
/// turn's request id and both are printed on every line the turn writes.
pub fn bind_span_to_trace(span: &tracing::Span, trace: &TurnTrace) {
    crate::otel_bridge::bind_parent(span, trace);
}

/// The `traceparent` to put on a call this turn makes to another process.
///
/// `None` when there is no trace to name, which is the case outside any turn
/// scope in a build with the `otel` feature off. A caller that gets `None`
/// injects nothing rather than a placeholder: a receiver that joins an invented
/// trace is worse than one that starts its own.
///
/// Which span is named as the parent depends on what this build can see. With
/// `otel` on the open span at the call site is named, so an MCP server's work
/// appears under the tool call that caused it. With `otel` off there is no span
/// id to read, so the turn's own root is named instead - the trace id is still
/// right, and the receiver's spans still land in the correct turn.
pub fn outbound_traceparent() -> Option<String> {
    #[cfg(feature = "otel")]
    if let Some(header) = crate::otel_bridge::current_span_traceparent() {
        return Some(header);
    }
    current_turn_trace().map(|trace| trace.root_header())
}

/// The 16 bytes a request id spells, or all zero when it spells nothing.
///
/// All zero is the W3C "no trace" sentinel, which
/// [`adelie_telemetry::resolve_trace_or_mint`] answers by generating an id, so
/// an unparseable request id degrades to a fresh trace rather than to none.
fn uuid_bytes(request_id: Option<&str>) -> [u8; 16] {
    request_id
        .and_then(|id| uuid::Uuid::parse_str(id).ok())
        .map_or([0; 16], uuid::Uuid::into_bytes)
}

/// The canonical rendering of a usable turn id, or `None`.
fn parse_turn_id(supplied: &str) -> Option<String> {
    let parsed = uuid::Uuid::parse_str(supplied).ok()?;
    if parsed.is_nil() {
        // The nil uuid spells the all-zero trace id, which the W3C spec
        // reserves as "invalid". A backend drops a span carrying it.
        return None;
    }
    Some(parsed.to_string())
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

    // -----------------------------------------------------------------
    // Adopting a client-minted turn id (epic D12).
    // -----------------------------------------------------------------

    const CLIENT_ID: &str = "11111111-2222-4333-8444-555555555555";

    #[test]
    fn client_supplied_turn_id_is_adopted_by_the_daemon() {
        assert_eq!(adopt_or_mint_turn_id(Some(CLIENT_ID)), CLIENT_ID);
    }

    #[test]
    fn missing_client_id_falls_back_to_daemon_minting() {
        let minted = adopt_or_mint_turn_id(None);
        let parsed = uuid::Uuid::parse_str(&minted).expect("the daemon mints a uuid");
        assert!(!parsed.is_nil(), "a minted id must be usable as a trace id");
    }

    #[test]
    fn malformed_or_nil_client_id_falls_back_to_minting() {
        // Each of these is a value a client can send today. None may fail the
        // turn, and none may become the trace id: the nil uuid spells the
        // all-zero trace id the W3C spec reserves as invalid, and a backend
        // drops a span carrying it.
        let unusable = [
            "",
            "not-a-uuid",
            "00000000-0000-0000-0000-000000000000",
            "11111111-2222-4333-8444-55555555555",
            "../../etc/passwd",
            "11111111-2222-4333-8444-555555555555-extra",
        ];
        for supplied in unusable {
            let answer = adopt_or_mint_turn_id(Some(supplied));
            assert_ne!(
                answer, supplied,
                "{supplied:?} is not usable and must not be adopted"
            );
            let parsed = uuid::Uuid::parse_str(&answer)
                .unwrap_or_else(|e| panic!("{supplied:?} produced {answer:?}, not a uuid: {e}"));
            assert!(!parsed.is_nil());
        }
    }

    #[test]
    fn an_adopted_turn_id_has_one_spelling() {
        // A client may spell a uuid without hyphens or in upper case. One turn
        // has one id, so the answer is the canonical rendering either way, and
        // the daemon returns it on the ack so the client learns which it is.
        let canonical = adopt_or_mint_turn_id(Some(CLIENT_ID));
        assert_eq!(
            adopt_or_mint_turn_id(Some("11111111222243338444555555555555")),
            canonical
        );
        assert_eq!(
            adopt_or_mint_turn_id(Some(
                "11111111-2222-4333-8444-555555555555"
                    .to_uppercase()
                    .as_str()
            )),
            canonical
        );
    }

    // -----------------------------------------------------------------
    // Resolving the trace.
    // -----------------------------------------------------------------

    /// A well-formed `traceparent` naming a trace no request id spells.
    const INCOMING: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const INCOMING_TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

    #[test]
    fn request_id_becomes_the_trace_id() {
        let trace = resolve_turn_trace(None, CLIENT_ID, "conv-1");
        assert_eq!(
            trace.trace.trace_id().to_hex(),
            CLIENT_ID.replace('-', ""),
            "a uuid and a trace id are both 16 bytes, so they are the same value"
        );
    }

    #[test]
    fn incoming_traceparent_is_continued_not_replaced() {
        let trace = resolve_turn_trace(Some(INCOMING), CLIENT_ID, "conv-1");
        assert_eq!(trace.trace.trace_id().to_hex(), INCOMING_TRACE_ID);
        assert!(
            trace.trace.parent_span_id().is_some(),
            "a continued trace carries the caller\'s span as the parent to hang from"
        );
    }

    #[test]
    fn a_malformed_traceparent_never_fails_the_turn() {
        // Oversized, wrong shape, reserved version, and the all-zero trace id.
        let long = format!("00-{}-00f067aa0ba902b7-01", "a".repeat(512));
        for header in [
            "garbage",
            "00-4bf92f3577b34da6a3ce929d0e0e4736",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            long.as_str(),
        ] {
            let trace = resolve_turn_trace(Some(header), CLIENT_ID, "conv-1");
            assert_eq!(
                trace.trace.trace_id().to_hex(),
                CLIENT_ID.replace('-', ""),
                "{header:?} is unusable, so the turn falls back to its own id"
            );
        }
    }

    #[test]
    fn a_turn_that_reached_the_loop_by_another_door_still_has_a_trace() {
        // An agent run, a scheduled job and a test carry no request id at all.
        // A generated trace id is still a trace; no trace is not.
        let trace = TurnTrace::minted(None, "conv-1");
        assert_ne!(trace.trace.trace_id().to_hex(), "0".repeat(32));
    }

    #[tokio::test]
    async fn the_outbound_traceparent_names_the_turns_trace() {
        let trace = resolve_turn_trace(None, CLIENT_ID, "conv-1");
        let header = with_turn_trace(Some(trace), async { outbound_traceparent() }).await;
        let header = header.expect("a turn in scope has a trace to name");
        assert!(
            header.contains(&CLIENT_ID.replace('-', "")),
            "the header must name this turn\'s trace: {header}"
        );
        assert_eq!(
            adelie_telemetry::extract_traceparent(&header)
                .expect("the header this crate writes must parse")
                .trace_id()
                .to_hex(),
            CLIENT_ID.replace('-', "")
        );
    }

    #[tokio::test]
    async fn there_is_no_outbound_traceparent_outside_a_turn() {
        assert_eq!(
            outbound_traceparent(),
            None,
            "a call made outside a turn must inject nothing rather than invent a trace"
        );
    }

    #[tokio::test]
    async fn the_conversation_id_is_readable_from_the_trace_scope() {
        let trace = resolve_turn_trace(None, CLIENT_ID, "conv-42");
        let seen = with_turn_trace(Some(trace), async { current_conversation_id() }).await;
        assert_eq!(seen, "conv-42");
        assert_eq!(
            current_conversation_id(),
            UNSET,
            "the slot must not leak past its scope"
        );
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
