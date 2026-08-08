//! What the `otel` feature adds: an exported span really carries the turn's
//! trace id, and an outbound call really names that trace.
//!
//! Everything else about a turn's telemetry is asserted with the feature off,
//! because that is the build every desktop install runs. These are the two
//! claims that cannot be: with `otel` off a span has no trace id at all, so a
//! test would be comparing an absence to itself.
//!
//! This target declares `required-features = ["otel"]`, so a default
//! `cargo test` skips it and resolves none of the opentelemetry crates.
//!
//! Run it with `cargo test -p desktop-assistant-core --features otel`.

use desktop_assistant_core::ports::turn_telemetry::{
    TurnTrace, bind_span_to_trace, outbound_traceparent, resolve_turn_trace, with_turn_trace,
};
use opentelemetry::trace::TracerProvider as _;
use tracing::Instrument;
use tracing_subscriber::layer::SubscriberExt;

/// The turn's correlation id.
const REQUEST_ID: &str = "11111111-2222-4333-8444-555555555555";

/// The same sixteen bytes as a W3C trace id.
const TRACE_ID: &str = "11111111222243338444555555555555";

const CONVERSATION_ID: &str = "conv-1";

/// A caller's own trace, whose id no request id here spells.
const INCOMING: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
const INCOMING_TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

/// Run `body` under a subscriber that gives every span a real trace context.
///
/// No exporter is attached. What is under test is the trace context a span
/// carries, and a provider assigns that whether or not anything ships it
/// anywhere.
fn with_otel_layer<T>(body: impl FnOnce() -> T) -> T {
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
    tracing::subscriber::with_default(subscriber, body)
}

/// The trace id an outbound call would name from inside `span`.
fn traceparent_inside(span: tracing::Span, trace: TurnTrace) -> String {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    runtime
        .block_on(with_turn_trace(Some(trace), async { outbound_traceparent() }).instrument(span))
        .expect("a span with a valid trace context names a traceparent")
}

#[test]
fn request_id_and_trace_id_are_the_same_value() {
    // The claim D12 rests on, in the build that exports. A uuid is 16 bytes
    // and a W3C trace id is 16 bytes, so the identifier in a user's report is
    // the identifier a backend indexes by, with no mapping table between them.
    let trace = resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID);
    let header = with_otel_layer(|| {
        let span = tracing::info_span!("turn");
        bind_span_to_trace(&span, &trace);
        traceparent_inside(span, trace.clone())
    });

    let parsed = adelie_telemetry::extract_traceparent(&header)
        .unwrap_or_else(|e| panic!("`{header}` is not a valid traceparent: {e}"));
    assert_eq!(
        parsed.trace_id().to_hex(),
        TRACE_ID,
        "the exported span must carry the trace id the request id spells"
    );
}

#[test]
fn client_turn_and_daemon_turn_share_one_trace() {
    // A turn started by a client and served by the daemon is one trace, not
    // two that happen to share a timestamp. The client sends the id; the
    // daemon's span is bound to it; an outbound call from inside that span
    // names the same trace. So the client, the daemon and the MCP server all
    // land in one place.
    let trace = resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID);
    let client_header = trace.root_header();

    let daemon_header = with_otel_layer(|| {
        let span = tracing::info_span!("turn");
        bind_span_to_trace(&span, &trace);
        traceparent_inside(span, trace.clone())
    });

    let client_trace = adelie_telemetry::extract_traceparent(&client_header)
        .expect("the client's own header must parse")
        .trace_id()
        .to_hex();
    let daemon_trace = adelie_telemetry::extract_traceparent(&daemon_header)
        .expect("the daemon's header must parse")
        .trace_id()
        .to_hex();

    assert_eq!(
        client_trace, daemon_trace,
        "the client and the daemon must name one trace, or the turn is drawn \
         twice and neither half shows the other"
    );
    assert_eq!(client_trace, TRACE_ID);
}

#[test]
fn incoming_traceparent_is_continued_not_replaced() {
    // A caller that already has a trace is joined. Without this the web BFF's
    // hop starts a second trace and the browser's half is lost.
    let trace = resolve_turn_trace(Some(INCOMING), REQUEST_ID, CONVERSATION_ID);
    let header = with_otel_layer(|| {
        let span = tracing::info_span!("turn");
        bind_span_to_trace(&span, &trace);
        traceparent_inside(span, trace.clone())
    });

    assert_eq!(
        adelie_telemetry::extract_traceparent(&header)
            .expect("the header must parse")
            .trace_id()
            .to_hex(),
        INCOMING_TRACE_ID,
        "the daemon must join the caller's trace rather than mint its own"
    );
}

#[test]
fn an_outbound_call_names_the_span_it_was_made_from() {
    // With `otel` on the open span is named as the parent, so an MCP server's
    // work appears under the tool call that caused it rather than flat under
    // the turn. The trace is the same either way; the shape is not.
    let trace = resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID);
    let (from_turn, from_tool) = with_otel_layer(|| {
        let turn = tracing::info_span!("turn");
        bind_span_to_trace(&turn, &trace);
        let from_turn = traceparent_inside(turn.clone(), trace.clone());
        let tool = tracing::info_span!(parent: &turn, "tool.call");
        let from_tool = traceparent_inside(tool, trace.clone());
        (from_turn, from_tool)
    });

    let span_id = |header: &str| {
        adelie_telemetry::extract_traceparent(header)
            .expect("the header must parse")
            .span_id()
            .to_hex()
    };
    assert_ne!(
        span_id(&from_turn),
        span_id(&from_tool),
        "a call from inside the tool span must name the tool span, or every \
         server's work hangs off the turn and the tool that caused it is lost"
    );
}
