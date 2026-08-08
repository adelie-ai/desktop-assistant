//! The two places a `tracing` span and an OpenTelemetry trace have to agree.
//!
//! `tracing` gives a span a numeric id that means nothing outside the process.
//! A trace backend needs the W3C pair - a 16-byte trace id and an 8-byte span
//! id - and `tracing-opentelemetry` is what puts one on the other. This module
//! is the only place in the crate that knows that, so the turn loop keeps
//! naming spans and nothing else learns about exporters.
//!
//! Both functions have a body for each build. With the `otel` feature off no
//! opentelemetry crate is resolved, [`bind_parent`] does nothing, and
//! [`current_span_traceparent`] answers `None`. Correlation still works in that
//! build, because the turn's trace id is its request id and both are printed on
//! every line the turn writes; what is missing is only the export.
//!
//! ## Why a turn span is given a parent that may not exist
//!
//! A turn's trace id is its request id, and the only way to make an exported
//! span carry a chosen trace id is to give it a parent that already has one.
//! So [`bind_parent`] builds a *remote* span context from the turn's trace and
//! sets it as the parent.
//!
//! When the trace was continued from a caller, that parent is real and the two
//! processes join up. When the trace was minted here, the parent names the
//! deterministic root id `adelie_telemetry::TraceParent::root_for` derives, and
//! that span is never exported by anybody - a client running a default-feature
//! build mints the id and exports nothing. A backend shows such a span as the
//! root of its trace, which is what it is. The alternative is a span with a
//! trace id nobody chose, and then the identifier in the user's report finds
//! nothing.

#[cfg(feature = "otel")]
use crate::ports::turn_telemetry::TurnTrace;

/// Make `span` carry this turn's trace id.
///
/// Does nothing with the `otel` feature off, where a span has no trace id to
/// carry.
#[cfg(feature = "otel")]
pub(crate) fn bind_parent(span: &tracing::Span, trace: &TurnTrace) {
    use adelie_telemetry::trace_context::TraceParent;
    use opentelemetry::trace::{TraceContextExt, TraceFlags, TraceState};
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let origin = trace.trace;
    let parent_span = origin
        .parent_span_id()
        .unwrap_or_else(|| TraceParent::root_for(origin.trace_id(), true).span_id());

    let context = opentelemetry::trace::SpanContext::new(
        opentelemetry::trace::TraceId::from_bytes(origin.trace_id().to_bytes()),
        opentelemetry::trace::SpanId::from_bytes(parent_span.to_bytes()),
        if trace.sampled() {
            TraceFlags::SAMPLED
        } else {
            TraceFlags::default()
        },
        // Remote: this context describes a span another process owns, which is
        // what stops the exporter reporting it as one of ours.
        true,
        TraceState::default(),
    );
    // Three things stop a parent being set, and only one of them is a defect
    // here. `AlreadyStarted` means a call site built the span, entered it, and
    // bound it afterwards, which is the wrong order. `SpanDisabled` is
    // ordinary: a daemon run at `RUST_LOG=warn` disables the INFO turn span,
    // and there is then no span to give a trace to. `LayerNotFound` means this
    // process installed no OpenTelemetry layer, which a foreign subscriber
    // already in place produces. So the error names itself rather than being
    // asserted, and the level is DEBUG because two of the three are normal.
    if let Err(error) =
        span.set_parent(opentelemetry::Context::new().with_remote_span_context(context))
    {
        tracing::debug!(%error, "the turn span keeps whatever trace it had");
    }
}

/// Make `span` carry this turn's trace id. A no-op in this build.
#[cfg(not(feature = "otel"))]
pub(crate) fn bind_parent(_span: &tracing::Span, _trace: &crate::ports::turn_telemetry::TurnTrace) {
}

/// The `traceparent` naming the span that is open right now.
///
/// `None` when no span is open, or when the span has no valid trace context -
/// which is what a build with the `otel` feature on but no subscriber layer
/// installed produces, and it must not be reported as a real trace.
#[cfg(feature = "otel")]
pub(crate) fn current_span_traceparent() -> Option<String> {
    use adelie_telemetry::trace_context::{SpanId, TraceId, TraceParent};
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let context = tracing::Span::current().context();
    let span_context = context.span().span_context().clone();
    if !span_context.is_valid() {
        return None;
    }
    let trace_id = TraceId::from_bytes(span_context.trace_id().to_bytes()).ok()?;
    let span_id = SpanId::from_bytes(span_context.span_id().to_bytes()).ok()?;
    Some(TraceParent::new(trace_id, span_id, span_context.is_sampled()).to_header())
}
