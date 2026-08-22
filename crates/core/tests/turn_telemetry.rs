//! What a turn says about itself: spans, metrics, and one completion line.
//!
//! An operator takes one identifier from a user report - "it took four minutes
//! at 14:20" - and follows that turn from the client, through the daemon,
//! through every tool round, to the provider call that was slow. These tests
//! are what that promise reduces to in this crate.
//!
//! ## Two kinds of assertion, and why both are needed
//!
//! A span records its fields when it is created, and **nothing prints them**
//! unless an event fires inside the span or the subscriber emits span-close
//! events. So a value captured into a span field is invisible to a test that
//! reads console text, and still exports over OTLP when the span closes. A
//! console assertion therefore proves something narrower than it appears: that
//! no *event* carries the value.
//!
//! Every test here that cares about span fields reads them back in process,
//! through [`SpanCapture`]. Every test that cares about a log line reads the
//! console text. Neither substitutes for the other.
//!
//! ## The registry is process-global
//!
//! `adelie_telemetry`'s metrics registry is a process-wide singleton and
//! `cargo test` runs this file's tests in one process, in parallel. Two turns
//! recording the same instrument at once make a count assertion fail
//! intermittently. [`METRICS`] serialises every test in this file for that
//! reason; assertions are on the delta across the turn, never on an absolute
//! total, because an earlier test in the same process has already recorded.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, Once};

use adelie_telemetry::metrics::{Label, Summary};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, Message, Role, ToolCall, ToolDefinition,
    ToolNamespace,
};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ReasoningConfig, TokenUsage,
};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::ports::tools::ToolExecutor;
use desktop_assistant_core::ports::turn_telemetry::{
    TurnRoute, TurnTrace, resolve_turn_trace, with_request_id, with_turn_route, with_turn_trace,
};
use desktop_assistant_core::service::ConversationHandler;
use tokio_util::sync::CancellationToken;
use tracing::Level;
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

// ---------------------------------------------------------------------------
// Fixtures. Each sentinel appears in exactly one place, so a failure names
// which content-bearing input leaked rather than only that something did.
// ---------------------------------------------------------------------------

/// The user's own words. Reaches the turn as the prompt.
const PROMPT_SENTINEL: &str = "SENTINEL-PROMPT-MY-DIAGNOSIS-AND-MY-ADDRESS";

/// A tool call's arguments. Chosen by the model, quoting the user.
const TOOL_ARGUMENT_SENTINEL: &str = "SENTINEL-ARGUMENT-sk-live-AND-A-HOME-PATH";

/// The model's own reply text.
const REPLY_SENTINEL: &str = "SENTINEL-REPLY-WHAT-THE-MODEL-CONCLUDED";

/// What a tool returned. The largest content surface in the loop: a file, a
/// web page, a database row.
const TOOL_RESULT_SENTINEL: &str = "SENTINEL-RESULT-THE-FILE-THIS-TOOL-READ";

/// What a failing tool said it could not do. Shaped like the real thing,
/// which quotes the path or the command it was given.
const TOOL_ERROR_SENTINEL: &str =
    "failed to read /home/example/.ssh/SENTINEL-ERROR-ED25519: permission denied";

/// The conversation id used by turns that check label bounding. Distinctive
/// enough that finding it in a label value is unambiguous.
const CONVERSATION_ID: &str = "conv-SENTINEL-UNBOUNDED-CONVERSATION-ID";

/// The user id used by the same turns, for the same reason.
const USER_ID: &str = "user-SENTINEL-UNBOUNDED-USER-ID";

/// The correlation id the transport would have minted.
const REQUEST_ID: &str = "11111111-2222-4333-8444-555555555555";

/// What a provider calls the call it just served. An id, not content, and the
/// only thing an LLM boundary gives back that is worth quoting to a provider.
const PROVIDER_REQUEST_ID: &str = "req_ExAmPlE0123456789";

/// The four OpenTelemetry GenAI attribute names a provider call reports its
/// token counts under.
///
/// Spelled out here rather than imported from the crate under test, so a
/// rename at the recording site fails these tests instead of travelling with
/// them. The names are the convention's, not this project's: see the constants
/// beside them in `core::telemetry` for the version they follow.
const GEN_AI_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
const GEN_AI_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
const GEN_AI_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";
const GEN_AI_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

/// All four, for the tests that assert on absence.
const GEN_AI_TOKEN_ATTRIBUTES: [&str; 4] = [
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OUTPUT_TOKENS,
    GEN_AI_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_CACHE_READ_INPUT_TOKENS,
];

const CONNECTION_ID: &str = "conn-primary";
const PROVIDER: &str = "example-connector";
const MODEL: &str = "example-model-v1";

fn route() -> TurnRoute {
    TurnRoute {
        connection_id: Some(CONNECTION_ID.to_string()),
        provider: Some(PROVIDER.to_string()),
        model: Some(MODEL.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Serialising the process-global metrics registry.
// ---------------------------------------------------------------------------

/// Serialises every test in this file.
///
/// `adelie_telemetry`'s registry is a process-wide singleton, and these tests
/// run in one process in parallel by default. Without this, two turns record
/// the same counter at once and a delta assertion fails with a count that is
/// right for neither test. Poisoning is ignored: a panic in one test must fail
/// that test, not cascade into every later one.
static METRICS: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
    METRICS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counter totals by (name, sorted labels), for delta comparison.
fn counters(summary: &Summary) -> HashMap<(String, String), u64> {
    summary
        .counters
        .iter()
        .map(|c| ((c.name.to_string(), render_labels(&c.labels)), c.total))
        .collect()
}

/// Histogram measurement counts by (name, sorted labels).
fn histograms(summary: &Summary) -> HashMap<(String, String), u64> {
    summary
        .histograms
        .iter()
        .map(|h| {
            (
                (h.name.to_string(), render_labels(&h.labels)),
                h.total.count,
            )
        })
        .collect()
}

fn render_labels(labels: &[Label]) -> String {
    labels
        .iter()
        .map(|l| format!("{}={}", l.key(), l.value()))
        .collect::<Vec<_>>()
        .join(",")
}

/// What one metric's counter rose by across a turn, summed over label sets
/// whose rendering contains every fragment in `label_contains`.
fn counter_delta(before: &Summary, after: &Summary, name: &str, label_contains: &[&str]) -> u64 {
    let before = counters(before);
    let mut delta = 0;
    for ((series, labels), total) in counters(after) {
        if series != name {
            continue;
        }
        if !label_contains.iter().all(|f| labels.contains(f)) {
            continue;
        }
        delta += total - before.get(&(series, labels)).copied().unwrap_or(0);
    }
    delta
}

/// How many measurements one histogram gained across a turn.
fn histogram_delta(before: &Summary, after: &Summary, name: &str, label_contains: &[&str]) -> u64 {
    let before = histograms(before);
    let mut delta = 0;
    for ((series, labels), count) in histograms(after) {
        if series != name {
            continue;
        }
        if !label_contains.iter().all(|f| labels.contains(f)) {
            continue;
        }
        delta += count - before.get(&(series, labels)).copied().unwrap_or(0);
    }
    delta
}

/// Every label value recorded during the window that just closed.
///
/// Read from the *window* rather than the total, so a value another test in
/// this process recorded earlier cannot be mistaken for one this turn wrote.
fn label_values_in_window(summary: &Summary) -> Vec<String> {
    let counters = summary
        .counters
        .iter()
        .filter(|c| c.window_delta > 0)
        .flat_map(|c| c.labels.iter());
    let histograms = summary
        .histograms
        .iter()
        .filter(|h| h.window.count > 0)
        .flat_map(|h| h.labels.iter());
    counters
        .chain(histograms)
        .map(|l| l.value().to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Reading spans back in process.
// ---------------------------------------------------------------------------

/// One span, as the subscriber saw it.
#[derive(Clone, Debug)]
struct SpanRecord {
    id: Id,
    name: &'static str,
    parent: Option<Id>,
    fields: HashMap<String, String>,
    /// When the span was created.
    opened: std::time::Instant,
    /// How long it stayed open, once it closed. `None` while it is still open.
    lifetime: Option<std::time::Duration>,
    /// Where the close fell in the run's sequence of spans and events. `None`
    /// while the span is still open.
    ///
    /// This is the half the harness was missing, and it is a sequence rather
    /// than a duration on purpose. A span's *extent* is what an exported trace
    /// draws, and it is decided by where the last handle drops - which is not
    /// where the code measuring the same work sits. Comparing the two by
    /// elapsed time needs the work between them to be slow enough to see, so
    /// the assertion would be a race. Comparing *order* against a log line the
    /// later work writes is exact.
    closed_at: Option<usize>,
}

impl SpanRecord {
    fn field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }
}

/// Everything one run's subscriber saw, in order.
#[derive(Default)]
struct Seen {
    spans: Vec<SpanRecord>,
    /// `(sequence, message)` for each event, so a span's close can be placed
    /// against the lines written before and after it.
    events: Vec<(usize, String)>,
    /// Monotonic across both, which is what makes the ordering comparable.
    next_seq: usize,
}

impl Seen {
    fn tick(&mut self) -> usize {
        self.next_seq += 1;
        self.next_seq
    }
}

/// A `tracing` layer that keeps every span it sees - fields, parent, and when
/// it closed relative to the events around it - so a test can read back both
/// what a span recorded and how far it reached.
#[derive(Clone, Default)]
struct SpanCapture(Arc<Mutex<Seen>>);

impl SpanCapture {
    fn seen(&self) -> (Vec<SpanRecord>, Vec<(usize, String)>) {
        let seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        (seen.spans.clone(), seen.events.clone())
    }
}

/// Renders each field value the way an exporter would read it.
struct FieldVisitor<'a>(&'a mut HashMap<String, String>);

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

impl<S> Layer<S> for SpanCapture
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = HashMap::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        // The registry establishes the parent before any layer's `on_new_span`
        // runs, so asking it is more reliable than reading the attributes -
        // which carry a parent only when the call site named one explicitly.
        let parent = ctx.span(id).and_then(|s| s.parent().map(|p| p.id()));
        let mut seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        seen.spans.push(SpanRecord {
            id: id.clone(),
            name: attrs.metadata().name(),
            parent,
            fields,
            opened: std::time::Instant::now(),
            lifetime: None,
            closed_at: None,
        });
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = HashMap::new();
        event.record(&mut FieldVisitor(&mut fields));
        let message = fields.remove("message").unwrap_or_default();
        let mut seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let seq = seen.tick();
        seen.events.push((seq, message));
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // Newest first: span ids are reused once a span closes, so an older
        // closed span can share this id.
        if let Some(span) = seen.spans.iter_mut().rev().find(|s| s.id == *id) {
            values.record(&mut FieldVisitor(&mut span.fields));
        }
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        let closed = std::time::Instant::now();
        let mut seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let seq = seen.tick();
        // Newest still-open span with this id: ids are reused after a close.
        if let Some(span) = seen
            .spans
            .iter_mut()
            .rev()
            .find(|s| s.id == id && s.closed_at.is_none())
        {
            span.lifetime = Some(closed.saturating_duration_since(span.opened));
            span.closed_at = Some(seq);
        }
    }
}

/// Everything one captured run produced.
struct Captured {
    console: String,
    spans: Vec<SpanRecord>,
    /// `(sequence, message)` for each event, in order, interleaved with the
    /// span closes recorded on [`SpanRecord::closed_at`].
    events: Vec<(usize, String)>,
    before: Summary,
    after: Summary,
}

impl Captured {
    fn span(&self, name: &str) -> &SpanRecord {
        self.spans
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "no `{name}` span was opened; the run produced {:?}",
                    self.span_names()
                )
            })
    }

    fn spans_named(&self, name: &str) -> Vec<&SpanRecord> {
        self.spans.iter().filter(|s| s.name == name).collect()
    }

    fn span_names(&self) -> Vec<&'static str> {
        self.spans.iter().map(|s| s.name).collect()
    }

    /// Where the first span of this name closed in the run's sequence.
    fn closed_at(&self, name: &str) -> usize {
        let spans = self.spans_named(name);
        assert!(
            !spans.is_empty(),
            "no `{name}` span opened, so there is no close to place"
        );
        spans[0]
            .closed_at
            .unwrap_or_else(|| panic!("a `{name}` span never closed"))
    }

    /// Where the first event whose message contains `needle` fell in the run's
    /// sequence.
    fn event_at(&self, needle: &str) -> usize {
        self.events
            .iter()
            .find(|(_, message)| message.contains(needle))
            .map(|(seq, _)| *seq)
            .unwrap_or_else(|| {
                panic!(
                    "no event said {needle:?}; the run wrote {:?}",
                    self.events
                        .iter()
                        .map(|(_, m)| m.as_str())
                        .collect::<Vec<_>>()
                )
            })
    }

    fn counter_delta(&self, name: &str, label_contains: &[&str]) -> u64 {
        counter_delta(&self.before, &self.after, name, label_contains)
    }

    fn histogram_delta(&self, name: &str, label_contains: &[&str]) -> u64 {
        histogram_delta(&self.before, &self.after, name, label_contains)
    }
}

// ---------------------------------------------------------------------------
// Console capture.
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .expect("captured log output is UTF-8")
    }
}

impl io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).extend(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

static PERMISSIVE_GLOBAL_DEFAULT: Once = Once::new();

/// Install one process-wide subscriber that accepts everything.
///
/// `tracing` caches each call site's interest globally, not per thread.
/// Without a permissive global default, a call site first evaluated while the
/// INFO-capped test holds the thread can latch "never" for the whole process,
/// and the TRACE-capped test then never sees it - a scheduling-dependent
/// flake, not a real failure.
fn ensure_permissive_global_default() {
    PERMISSIVE_GLOBAL_DEFAULT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::TRACE)
            .with_writer(io::sink)
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("install the permissive global default exactly once");
    });
}

/// Drive `future` with the console captured at `level`, every span read back
/// in process, and the metrics registry sampled either side.
///
/// A current-thread runtime keeps the whole run on the thread holding the
/// thread-local subscriber, so the capture covers the turn and not only its
/// first poll.
fn capture<F: Future<Output = ()>>(level: Level, future: F) -> Captured {
    ensure_permissive_global_default();
    let console = CapturedLog::default();
    let spans = SpanCapture::default();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_subscriber::filter::LevelFilter::from_level(level))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(console.clone())
                .with_ansi(false)
                .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE),
        )
        .with(spans.clone());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a current-thread runtime");

    // Close the registry's window first, so `window_delta` and `window.count`
    // report only what this turn recorded.
    adelie_telemetry::metrics::global().dump_now();
    let before = adelie_telemetry::metrics::global().snapshot();
    subscriber.set_default_and(|| runtime.block_on(future));
    let after = adelie_telemetry::metrics::global().dump_now();

    let (spans, events) = spans.seen();
    Captured {
        console: console.text(),
        spans,
        events,
        before,
        after,
    }
}

/// `tracing_subscriber` has no combinator for "make this the default for the
/// duration of a closure" on a layered subscriber, so this names the two-step.
trait SetDefaultAnd: Sized {
    fn set_default_and<R>(self, body: impl FnOnce() -> R) -> R;
}

impl<S> SetDefaultAnd for S
where
    S: tracing::Subscriber + Send + Sync + 'static,
{
    fn set_default_and<R>(self, body: impl FnOnce() -> R) -> R {
        let _guard = tracing::subscriber::set_default(self);
        body()
    }
}

// ---------------------------------------------------------------------------
// Stubs. The turn has to be real; nothing it talks to does.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MemStore {
    data: Mutex<HashMap<String, Conversation>>,
}

impl ConversationStore for MemStore {
    async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
        self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
        Ok(())
    }

    async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.data
            .lock()
            .unwrap()
            .get(&id.0)
            .cloned()
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
    }

    async fn list(&self) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(self
            .data
            .lock()
            .unwrap()
            .values()
            .map(ConversationSummary::from)
            .collect())
    }

    async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
        self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
        Ok(())
    }

    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.data
            .lock()
            .unwrap()
            .remove(&id.0)
            .map(|_| ())
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
    }

    async fn archive(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }

    async fn unarchive(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }

    async fn create_summary(
        &self,
        _conversation_id: &ConversationId,
        _summary: String,
        _start_ordinal: usize,
        _end_ordinal: usize,
    ) -> Result<String, CoreError> {
        Ok("sum".into())
    }

    async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
        Ok(())
    }
}

/// One scripted provider turn: a reply, or a failure.
enum Reply {
    Answer(Box<LlmResponse>),
    Fail(CoreError),
}

impl From<LlmResponse> for Reply {
    fn from(response: LlmResponse) -> Self {
        Self::Answer(Box::new(response))
    }
}

/// An LLM that replays a script, so the turn's shape is fixed by the test.
struct ScriptedLlm {
    responses: Mutex<Vec<Reply>>,
    /// What this connector says the provider called the call. A `&'static str`
    /// so a test can hand it a hostile value without the harness bounding it
    /// on the way in - the bounding under test happens at the recording site.
    provider_request_id: &'static str,
}

impl ScriptedLlm {
    fn new(responses: Vec<Reply>) -> Self {
        Self {
            responses: Mutex::new(responses),
            provider_request_id: PROVIDER_REQUEST_ID,
        }
    }

    /// A connector whose provider answers with `id` as its request identifier.
    fn reporting(responses: Vec<Reply>, id: &'static str) -> Self {
        Self {
            responses: Mutex::new(responses),
            provider_request_id: id,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for ScriptedLlm {
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: &[ToolDefinition],
        _reasoning: ReasoningConfig,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // What a real connector does the moment the provider answers, before
        // anything about the body is known: report the provider's own
        // identifier onto the open `llm.call` span. No provider continues our
        // trace, so capturing that id is the only thing this boundary allows,
        // and it matters most on the calls that fail.
        desktop_assistant_core::ports::llm::record_provider_request_id(self.provider_request_id);
        let reply = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Ok(LlmResponse::text("done"));
            }
            responses.remove(0)
        };
        let response = match reply {
            Reply::Answer(response) => *response,
            Reply::Fail(error) => return Err(error),
        };
        if !response.text.is_empty() {
            on_chunk(response.text.clone());
        }
        Ok(response)
    }
}

struct ScriptedTools {
    tools: Vec<ToolDefinition>,
    /// When set, every dispatch fails with this message instead of succeeding.
    failure: Option<String>,
    /// When set, the first dispatch trips the turn's cancellation token, so
    /// the loop's per-tool checkpoint fires on the next call. Nothing else
    /// reaches that exit: a token the test trips before the turn starts stops
    /// it before any tool runs.
    cancel_after_first: bool,
    /// Tools the executor knows and will run, but does **not** return from
    /// `core_tools`. This is the real fleet's shape: the daemon's executor
    /// answers with builtins only, and an MCP tool reaches a turn's tool list
    /// through per-turn activation. So a fleet tool the model learned about in
    /// an earlier turn is one this turn never offered and still executes.
    unadvertised: Vec<ToolDefinition>,
    /// When set, the tool returns a multi-megabyte payload, so the loop's
    /// post-measurement truncation costs measurable time.
    oversized_output: bool,
    /// When set, every dispatch answers with a tool-search result naming this
    /// tool, so the turn loop activates it for the rounds that follow.
    search_hit: Option<&'static str>,
}

impl ScriptedTools {
    fn ok() -> Self {
        Self {
            tools: vec![write_note()],
            failure: None,
            cancel_after_first: false,
            unadvertised: Vec::new(),
            oversized_output: false,
            search_hit: None,
        }
    }

    /// An executor that knows `name` and will run it, without ever offering it.
    fn knowing(name: &str) -> Self {
        Self {
            tools: vec![write_note()],
            failure: None,
            cancel_after_first: false,
            unadvertised: vec![ToolDefinition::new(
                name,
                "a fleet tool this turn never offered",
                serde_json::json!({"type": "object"}),
            )],
            oversized_output: false,
            search_hit: None,
        }
    }

    fn failing() -> Self {
        Self {
            tools: vec![write_note()],
            failure: Some(TOOL_ERROR_SENTINEL.to_string()),
            cancel_after_first: false,
            unadvertised: Vec::new(),
            oversized_output: false,
            search_hit: None,
        }
    }

    /// A tool whose output is large enough that capping it afterwards costs
    /// real time, which is what separates a span from its own measurement.
    fn oversized() -> Self {
        Self {
            tools: vec![write_note()],
            failure: None,
            cancel_after_first: false,
            unadvertised: Vec::new(),
            oversized_output: true,
            search_hit: None,
        }
    }

    /// An executor whose one advertised tool is the discovery tool, and whose
    /// search hands back one fleet tool. The turn loop activates it, so round
    /// two advertises a strictly larger block than round one - the growth
    /// #1212 bounds, and the only shape that tells a peak apart from an
    /// opening figure.
    fn searching() -> Self {
        Self {
            tools: vec![tool_search()],
            failure: None,
            cancel_after_first: false,
            unadvertised: vec![ToolDefinition::new(
                FLEET_TOOL,
                "a fleet tool reached only through a search",
                serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            )],
            oversized_output: false,
            search_hit: Some(FLEET_TOOL),
        }
    }

    /// A tool that succeeds and then cancels the turn, so the loop's per-tool
    /// cancellation checkpoint is the exit taken.
    fn cancelling() -> Self {
        Self {
            tools: vec![write_note()],
            failure: None,
            cancel_after_first: true,
            unadvertised: Vec::new(),
            oversized_output: false,
            search_hit: None,
        }
    }
}

/// The name of the fleet tool [`ScriptedTools::searching`] hands back.
const FLEET_TOOL: &str = "fleet_lookup";

/// The daemon's discovery tool, by the name the turn loop watches for.
fn tool_search() -> ToolDefinition {
    ToolDefinition::new(
        "builtin_tool_search",
        "find tools",
        serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
    )
}

/// One search round then an answer, so round two advertises what round one's
/// search activated.
fn search_script() -> Vec<Reply> {
    vec![
        LlmResponse::with_tool_calls(
            "",
            vec![ToolCall::new(
                "s1",
                "daemon_builtin_tool_search",
                r#"{"query":"anything"}"#,
            )],
        )
        .with_usage(usage(100, 10))
        .into(),
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(200, 20))
            .into(),
    ]
}

fn write_note() -> ToolDefinition {
    ToolDefinition::new(
        "write_note",
        "write a note",
        serde_json::json!({"type": "object"}),
    )
}

impl ToolExecutor for ScriptedTools {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        self.tools.clone()
    }

    async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        Ok(vec![])
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        Ok(self
            .tools
            .iter()
            .chain(self.unadvertised.iter())
            .find(|t| t.name == name)
            .cloned())
    }

    async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        Vec::new()
    }

    async fn execute_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        if self.cancel_after_first {
            desktop_assistant_core::ports::llm::current_cancellation_token()
                .expect("the turn installs a cancellation token")
                .cancel();
        }
        if self.oversized_output {
            return Ok("x".repeat(OVERSIZED_TOOL_RESULT_BYTES));
        }
        if let Some(hit) = self.search_hit {
            return Ok(serde_json::json!({
                "ok": true,
                "tools": [{
                    "name": format!("daemon_{hit}"),
                    "description": "a fleet tool",
                    "runs_on": "daemon",
                }],
            })
            .to_string());
        }
        match &self.failure {
            Some(message) => Err(CoreError::ToolExecution(message.clone())),
            // The tool's own output is content and belongs at DEBUG. Returning
            // a sentinel rather than "ok" is what lets the leak tests see the
            // largest content surface the loop handles.
            None => Ok(TOOL_RESULT_SENTINEL.to_string()),
        }
    }
}

fn handler(
    responses: Vec<Reply>,
    tools: ScriptedTools,
) -> ConversationHandler<MemStore, ScriptedLlm, ScriptedTools> {
    ConversationHandler::with_tools(
        MemStore::default(),
        ScriptedLlm::new(responses),
        tools,
        Box::new(|| CONVERSATION_ID.to_string()),
    )
}

/// What a provider that does no prompt caching reports: two counts, and
/// nothing for either cache.
fn usage(input: u64, output: u64) -> TokenUsage {
    TokenUsage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    }
}

/// What a caching provider reports: all four counts.
fn caching_usage(input: u64, output: u64, cache_write: u64, cache_read: u64) -> TokenUsage {
    TokenUsage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        cache_creation_input_tokens: Some(cache_write),
        cache_read_input_tokens: Some(cache_read),
    }
}

fn tool_call(id: &str) -> ToolCall {
    ToolCall::new(
        id,
        "write_note",
        serde_json::json!({ "note": TOOL_ARGUMENT_SENTINEL }).to_string(),
    )
}

/// Two tool rounds then an answer, each round reporting its own usage.
fn three_round_script() -> Vec<Reply> {
    vec![
        LlmResponse::with_tool_calls("", vec![tool_call("c1")])
            .with_usage(usage(100, 10))
            .into(),
        LlmResponse::with_tool_calls("", vec![tool_call("c2")])
            .with_usage(usage(200, 20))
            .into(),
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(300, 30))
            .into(),
    ]
}

/// One tool round then an answer.
fn two_round_script() -> Vec<Reply> {
    vec![
        LlmResponse::with_tool_calls("", vec![tool_call("c1")])
            .with_usage(usage(100, 10))
            .into(),
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(200, 20))
            .into(),
    ]
}

/// Run one turn, with the request id, the route and the user id installed the
/// way the daemon installs them.
async fn one_turn(handler: &ConversationHandler<MemStore, ScriptedLlm, ScriptedTools>) {
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    with_user_id(
        UserId::new(USER_ID),
        with_request_id(
            REQUEST_ID.to_string(),
            with_turn_route(route(), async {
                // The result is deliberately dropped. Half the paths under
                // test end in an error, and what they are being tested for is
                // what they recorded on the way out.
                let _ = handler
                    .send_prompt_with_override(
                        &conv.id,
                        PROMPT_SENTINEL.to_string(),
                        None,
                        String::new(),
                        Box::new(|_| true),
                        Box::new(|_| {}),
                        CancellationToken::new(),
                    )
                    .await;
            }),
        ),
    )
    .await;
}

/// Run one turn with the trace the transport would have resolved installed,
/// the way the dispatcher installs it before the turn body is spawned.
async fn one_turn_traced(
    handler: &ConversationHandler<MemStore, ScriptedLlm, ScriptedTools>,
    trace: TurnTrace,
) {
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    with_turn_trace(
        Some(trace),
        with_user_id(
            UserId::new(USER_ID),
            with_request_id(
                REQUEST_ID.to_string(),
                with_turn_route(route(), async {
                    let _ = handler
                        .send_prompt_with_override(
                            &conv.id,
                            PROMPT_SENTINEL.to_string(),
                            None,
                            String::new(),
                            Box::new(|_| true),
                            Box::new(|_| {}),
                            CancellationToken::new(),
                        )
                        .await;
                }),
            ),
        ),
    )
    .await;
}

fn run(level: Level, responses: Vec<Reply>, tools: ScriptedTools) -> Captured {
    capture(level, async move {
        let handler = handler(responses, tools);
        one_turn(&handler).await;
    })
}

// ---------------------------------------------------------------------------
// Spans and the correlation id.
// ---------------------------------------------------------------------------

#[test]
fn turn_emits_a_span_with_the_correlation_ids() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    assert_eq!(
        turn.field("request_id"),
        Some(REQUEST_ID),
        "the turn span must carry the id the client correlates its stream by; \
         got {:?}",
        turn.fields
    );
    assert_eq!(
        turn.field("conversation_id"),
        Some(CONVERSATION_ID),
        "got {:?}",
        turn.fields
    );
    assert_eq!(
        turn.field("user_id"),
        Some(USER_ID),
        "got {:?}",
        turn.fields
    );
    assert_eq!(
        turn.field("connection_id"),
        Some(CONNECTION_ID),
        "got {:?}",
        turn.fields
    );
    assert_eq!(turn.field("model"), Some(MODEL), "got {:?}", turn.fields);
}

#[test]
fn each_tool_round_is_a_child_span_of_the_turn() {
    let _serialised = serialised();
    let captured = run(Level::INFO, three_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");
    let rounds = captured.spans_named("turn.round");

    assert_eq!(
        rounds.len(),
        3,
        "a three-round turn opens three round spans; got {:?}",
        captured.span_names()
    );
    for (index, round) in rounds.iter().enumerate() {
        assert_eq!(
            round.parent.as_ref(),
            Some(&turn.id),
            "round {index} must hang from the turn span, not from the root"
        );
        assert_eq!(
            round.field("round"),
            Some((index + 1).to_string().as_str()),
            "each round span names its own index, one-based, so a trace reads \
             the same way as the log line beside it"
        );
    }
}

#[test]
fn the_llm_call_and_each_tool_dispatch_are_children_of_their_round() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let first_round = captured.spans_named("turn.round")[0].id.clone();

    let llm = captured
        .spans_named("llm.call")
        .into_iter()
        .find(|s| s.parent.as_ref() == Some(&first_round))
        .expect("the first round's LLM call opens a span under that round");
    assert_eq!(llm.field("model"), Some(MODEL), "got {:?}", llm.fields);

    let tool = captured
        .spans_named("tool.call")
        .into_iter()
        .find(|s| s.parent.as_ref() == Some(&first_round))
        .expect("the tool the first round dispatched opens a span under it");
    assert_eq!(
        tool.field("tool"),
        Some("write_note"),
        "the tool span names which tool ran; got {:?}",
        tool.fields
    );
}

#[test]
fn every_log_line_in_a_turn_carries_the_request_id() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let lines: Vec<&str> = captured
        .console
        .lines()
        .filter(|l| l.contains("INFO") || l.contains("WARN"))
        .collect();
    assert!(
        !lines.is_empty(),
        "the turn must write something at INFO for this test to mean anything\n\
         --- console ---\n{}",
        captured.console
    );
    for line in &lines {
        assert!(
            line.contains(REQUEST_ID),
            "every line written inside the turn must carry the correlation id \
             through span scope rather than by hand; this one did not:\n  {line}\n\
             --- console ---\n{}",
            captured.console
        );
    }
}

// ---------------------------------------------------------------------------
// The completion line.
// ---------------------------------------------------------------------------

#[test]
fn turn_completion_line_carries_duration_model_rounds_and_tokens() {
    let _serialised = serialised();
    let captured = run(Level::INFO, three_round_script(), ScriptedTools::ok());

    let line = captured
        .console
        .lines()
        .find(|l| l.contains("turn finished"))
        .unwrap_or_else(|| {
            panic!(
                "a turn must write exactly one completion line an operator can grep\n\
                 --- console ---\n{}",
                captured.console
            )
        })
        .to_string();

    for field in [
        "duration_ms=",
        &format!("model=\"{MODEL}\""),
        "rounds=3",
        "input_tokens=600",
        "output_tokens=60",
        "outcome=\"answered\"",
    ] {
        assert!(
            line.contains(field),
            "the completion line must carry `{field}` as a field, not interpolated \
             into the message:\n  {line}"
        );
    }
    assert_eq!(
        captured
            .console
            .lines()
            .filter(|l| l.contains("turn finished"))
            .count(),
        1,
        "one completion line per turn, not one per round"
    );
}

// ---------------------------------------------------------------------------
// Token usage: per round, all four counts, and `None` is not zero.
// ---------------------------------------------------------------------------

#[test]
fn token_usage_reaches_the_metrics_facade() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    assert_eq!(
        captured.counter_delta("llm.tokens.input", &[]),
        300,
        "both rounds' input tokens must reach the facade"
    );
    assert_eq!(
        captured.counter_delta("llm.tokens.output", &[]),
        30,
        "both rounds' output tokens must reach the facade"
    );
}

#[test]
fn token_usage_is_recorded_per_round() {
    let _serialised = serialised();
    let captured = run(Level::INFO, three_round_script(), ScriptedTools::ok());

    let rounds = captured.spans_named("turn.round");
    let recorded: Vec<Option<&str>> = rounds.iter().map(|r| r.field("input_tokens")).collect();
    assert_eq!(
        recorded,
        vec![Some("100"), Some("200"), Some("300")],
        "a turn of three rounds produces three separate recordings, not one \
         total: the interesting question is which round blew up"
    );
}

#[test]
fn missing_token_counts_are_not_recorded_as_zero() {
    let _serialised = serialised();
    // A connector that reports input but not output. Recording `0` for the
    // absence would silently understate every total that includes it, with no
    // way afterwards to tell a real zero from a missing number.
    let script = vec![
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(TokenUsage {
                input_tokens: Some(100),
                output_tokens: None,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            })
            .into(),
    ];
    let captured = run(Level::INFO, script, ScriptedTools::ok());

    assert_eq!(
        captured.counter_delta("llm.tokens.input", &[]),
        100,
        "the count that was reported must still be recorded"
    );
    assert_eq!(
        captured.counter_delta("llm.tokens.output", &[]),
        0,
        "a count the connector did not report contributes nothing to the total"
    );
    assert_eq!(
        captured.counter_delta("llm.tokens.unreported", &["count=output"]),
        1,
        "the absence is counted separately, so a total that looks low can be \
         checked against how many calls did not report"
    );
    assert_eq!(
        captured.counter_delta("llm.tokens.unreported", &["count=input"]),
        0,
        "a count that WAS reported must not be counted as unreported"
    );
    assert_eq!(
        captured.spans_named("turn.round")[0].field("output_tokens"),
        None,
        "an absent count must not appear on the span as a zero either"
    );
}

#[test]
fn cache_token_counts_are_recorded_when_present() {
    let _serialised = serialised();
    // On a caching provider the cache counts are the whole cost story: a cache
    // read costs a fraction of a fresh input token, so reporting input alone
    // makes a well-cached turn look identical to a cold one.
    let script = vec![
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(TokenUsage {
                input_tokens: Some(100),
                output_tokens: Some(10),
                cache_creation_input_tokens: Some(40),
                cache_read_input_tokens: Some(4_000),
            })
            .into(),
    ];
    let captured = run(Level::INFO, script, ScriptedTools::ok());

    assert_eq!(captured.counter_delta("llm.tokens.cache_write", &[]), 40);
    assert_eq!(captured.counter_delta("llm.tokens.cache_read", &[]), 4_000);

    let round = captured.spans_named("turn.round")[0];
    assert_eq!(round.field("cache_write_tokens"), Some("40"));
    assert_eq!(round.field("cache_read_tokens"), Some("4000"));
}

#[test]
fn token_counts_appear_at_info() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let line = captured
        .console
        .lines()
        .find(|l| l.contains("round finished") && l.contains("input_tokens=100"))
        .unwrap_or_else(|| {
            panic!(
                "each round writes its own counts at INFO, as structured fields\n\
                 --- console ---\n{}",
                captured.console
            )
        });
    assert!(
        line.contains("output_tokens=10"),
        "the same line carries the output count:\n  {line}"
    );
    assert!(
        line.contains("round=1"),
        "and says which round it is talking about:\n  {line}"
    );
}

#[test]
fn round_span_carries_token_attributes() {
    let _serialised = serialised();
    // Read back in process, not scraped from the console: a span field never
    // reaches the console unless an event fires inside the span, so a console
    // assertion cannot see this either way.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let round = captured.spans_named("turn.round")[0];

    assert_eq!(round.field("input_tokens"), Some("100"));
    assert_eq!(round.field("output_tokens"), Some("10"));
}

#[test]
fn token_metric_labels_are_bounded() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let values = label_values_in_window(&captured.after);
    assert!(
        !values.is_empty(),
        "the turn must have recorded something for this test to mean anything"
    );
    assert!(
        values.iter().any(|v| v == MODEL),
        "model is a useful axis and is bounded by configuration, so it must be \
         a label; recorded values were {values:?}"
    );
    assert!(
        values.iter().any(|v| v == PROVIDER),
        "so is provider; recorded values were {values:?}"
    );
    for unbounded in [CONVERSATION_ID, USER_ID, REQUEST_ID] {
        assert!(
            !values.iter().any(|v| v == unbounded),
            "`{unbounded}` is unbounded and would burn the 64-value cap on \
             first contact; recorded values were {values:?}"
        );
    }
}

#[test]
fn a_failed_round_still_records_what_it_knows() {
    let _serialised = serialised();
    // The model answered and the tool then failed. The tokens were spent
    // whatever happened next, so the error path has to reach the recording
    // site rather than dropping the round's numbers.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::failing());

    assert_eq!(
        captured.counter_delta("llm.tokens.input", &[]),
        300,
        "a round that fails after the model responded still consumed tokens"
    );
    let round = captured.spans_named("turn.round")[0];
    assert_eq!(round.field("input_tokens"), Some("100"));
    assert_eq!(
        round.field("outcome"),
        Some("tool_error"),
        "and the round says what went wrong; got {:?}",
        round.fields
    );
}

// ---------------------------------------------------------------------------
// Latency, which is the whole point.
// ---------------------------------------------------------------------------

#[test]
fn a_turn_decomposes_into_provider_time_and_tool_time() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    assert_eq!(
        captured.histogram_delta("turn.duration", &["outcome=answered"]),
        1,
        "one turn, one duration measurement"
    );
    assert_eq!(
        captured.histogram_delta("turn.round.duration", &[]),
        2,
        "two rounds, two round measurements"
    );
    assert_eq!(
        captured.histogram_delta(
            "llm.call.duration",
            &[
                &format!("model={MODEL}"),
                &format!("provider={PROVIDER}"),
                "purpose=turn",
            ]
        ),
        2,
        "each round's provider call is measured by provider and model, so a \
         slow provider is attributable without reproducing the turn"
    );
    assert_eq!(
        captured.histogram_delta("llm.call.duration", &["purpose=title"]),
        1,
        "and the provider time a turn spends outside its rounds - here naming \
         a new conversation - is measured too, separated by purpose, so a turn \
         whose minutes went into an overhead does not decompose into a gap"
    );
    assert_eq!(
        captured.histogram_delta("tool.call.duration", &["tool=write_note", "outcome=ok"]),
        1,
        "and each tool dispatch is measured by tool name and outcome"
    );
}

#[test]
fn a_failing_tool_is_measured_as_a_failure() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::failing());

    assert_eq!(
        captured.histogram_delta("tool.call.duration", &["tool=write_note", "outcome=error"]),
        1,
        "a tool that failed must not be counted as one that worked"
    );
    assert_eq!(
        captured.histogram_delta("tool.call.duration", &["tool=write_note", "outcome=ok"]),
        0,
    );
}

// ---------------------------------------------------------------------------
// A span must end with the work it measures.
// ---------------------------------------------------------------------------

/// Enough tool output to make the post-measurement truncation cost real time.
const OVERSIZED_TOOL_RESULT_BYTES: usize = 8 * 1024 * 1024;

/// A cap far below that, so `cap_tool_result` genuinely truncates.
const TOOL_RESULT_CAP_BYTES: usize = 4_096;

#[test]
fn each_instrumented_call_closes_its_span_with_its_own_measurement() {
    let _serialised = serialised();
    // A span's extent is decided by where its last handle drops, and the code
    // that measures the same work sits somewhere else. When the two part
    // company the histogram says one number and the exported trace draws
    // another - and only the trace is wrong, in the direction that blames
    // whatever the span is named after.
    //
    // Asserted as an *order*, not as a duration. The work that separates them
    // - capping an oversized tool result - is fast enough that the two
    // elapsed times overlap, so a timing assertion would be a race that passes
    // whichever way the code is written. Where the close falls against a line
    // the later work writes is exact.
    //
    // The tool arm needs an oversized result because the truncation line only
    // appears when there is something to truncate.
    let tools = ScriptedTools::oversized();
    let captured = capture(Level::INFO, async move {
        let handler = ConversationHandler::with_tools(
            MemStore::default(),
            ScriptedLlm::new(two_round_script()),
            tools,
            Box::new(|| CONVERSATION_ID.to_string()),
        )
        .with_max_tool_result_bytes(TOOL_RESULT_CAP_BYTES);
        one_turn(&handler).await;
    });

    // The tool span must close before the loop caps its result, which happens
    // after the measurement and costs more the larger the payload is.
    assert!(
        captured.closed_at("tool.call") < captured.event_at("ingestion cap"),
        "the tool span was still open while the loop capped the result, so it \
         reaches past the work its metric measured"
    );

    // The provider span must close before the loop reads the response it
    // returned, which is the first thing the round does with it.
    assert!(
        captured.closed_at("llm.call") < captured.event_at("LLM requested"),
        "the provider span was still open while the round processed the \
         response, so it reaches past the call its metric measured"
    );

    // And every instrumented span must actually close, or the two assertions
    // above are comparing against a span that is simply never accounted for.
    for name in ["turn", "turn.round", "llm.call", "tool.call"] {
        for span in captured.spans_named(name) {
            assert!(
                span.lifetime.is_some(),
                "a `{name}` span never closed, so nothing exports its duration"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The level contract (epic D10), on the new surface.
// ---------------------------------------------------------------------------

/// Every sentinel, so a failure names which content-bearing value leaked
/// rather than only that something did.
const SENTINELS: [(&str, &str); 5] = [
    ("the user's prompt", PROMPT_SENTINEL),
    ("a tool call's arguments", TOOL_ARGUMENT_SENTINEL),
    ("the model's reply", REPLY_SENTINEL),
    ("a tool's output", TOOL_RESULT_SENTINEL),
    ("a failing tool's message", TOOL_ERROR_SENTINEL),
];

/// One path a turn can take.
struct TurnPath {
    /// What the case is, phrased for a failure message.
    name: &'static str,
    script: Vec<Reply>,
    tools: ScriptedTools,
    /// The round outcome this path must produce. Asserted separately, so a
    /// "failure" that silently succeeded cannot make a leak test pass
    /// vacuously by never running the path it claims.
    outcome: &'static str,
    /// Whether a tool actually runs on this path. Two of the sentinels can
    /// only appear when one does, so without this a row that never dispatches
    /// looks like coverage of them and is not.
    runs_a_tool: bool,
}

/// Every path a turn can take, so the content assertion covers the failure
/// arms and not only the one that works.
fn every_turn_path() -> Vec<TurnPath> {
    vec![
        TurnPath {
            name: "the model answers after a tool round",
            script: two_round_script(),
            tools: ScriptedTools::ok(),
            outcome: "answered",
            runs_a_tool: true,
        },
        TurnPath {
            name: "a tool fails and the turn carries on",
            script: two_round_script(),
            tools: ScriptedTools::failing(),
            outcome: "tool_error",
            runs_a_tool: true,
        },
        TurnPath {
            name: "the provider call fails",
            script: vec![Reply::Fail(CoreError::Llm(
                "the provider is unreachable".to_string(),
            ))],
            tools: ScriptedTools::ok(),
            outcome: "llm_error",
            runs_a_tool: false,
        },
        TurnPath {
            name: "the provider call is cancelled",
            script: vec![Reply::Fail(CoreError::Cancelled)],
            tools: ScriptedTools::ok(),
            outcome: "cancelled",
            runs_a_tool: false,
        },
        TurnPath {
            name: "the user cancels between tool dispatches",
            // Two calls in one round, because the loop's per-tool checkpoint
            // runs at the top of each iteration: with a single call there is
            // no second iteration to reach it, and the cancellation would
            // instead be seen by the between-rounds check, which is a
            // different exit.
            script: vec![
                LlmResponse::with_tool_calls("", vec![tool_call("c1"), tool_call("c2")])
                    .with_usage(usage(100, 10))
                    .into(),
                LlmResponse::text(REPLY_SENTINEL)
                    .with_usage(usage(200, 20))
                    .into(),
            ],
            tools: ScriptedTools::cancelling(),
            outcome: "cancelled",
            runs_a_tool: true,
        },
    ]
}

#[test]
fn turn_span_records_no_content() {
    let _serialised = serialised();
    for TurnPath {
        name,
        script,
        tools,
        ..
    } in every_turn_path()
    {
        let captured = run(Level::INFO, script, tools);

        // Span fields first: an `#[instrument]` without `skip` captures its
        // arguments, and nothing prints them, so this is invisible on the
        // console and still exports over OTLP.
        for span in &captured.spans {
            for (key, value) in &span.fields {
                for (what, sentinel) in SENTINELS {
                    assert!(
                        !value.contains(sentinel),
                        "on the path where {name}: span `{}` field `{key}` \
                         carries {what}: {value}",
                        span.name
                    );
                }
            }
        }

        // Then events, which the console does show.
        for (what, sentinel) in SENTINELS {
            assert!(
                !captured.console.contains(sentinel),
                "on the path where {name}: {what} reached an INFO line\n\
                 --- console ---\n{}",
                captured.console
            );
        }
    }
}

#[test]
fn every_probed_path_really_took_the_path_it_names() {
    let _serialised = serialised();
    // A leak test that drives a "failure" which silently succeeds passes
    // vacuously - it asserts nothing leaked from a path that never ran. This
    // asserts each case in the table produced the round outcome it claims,
    // and that the rows claiming to run a tool really dispatched one, which is
    // what makes the tool-result and tool-error sentinels mean anything.
    for TurnPath {
        name,
        script,
        tools,
        outcome,
        runs_a_tool,
    } in every_turn_path()
    {
        let captured = run(Level::INFO, script, tools);
        let rounds = captured.spans_named("turn.round");
        let observed: Vec<Option<&str>> = rounds.iter().map(|r| r.field("outcome")).collect();
        assert!(
            observed.contains(&Some(outcome)),
            "the case where {name} must produce a round with outcome \
             `{outcome}`; got {observed:?}"
        );
        assert_eq!(
            !captured.spans_named("tool.call").is_empty(),
            runs_a_tool,
            "the case where {name} says runs_a_tool={runs_a_tool}, and the run \
             disagrees; the sentinels a tool produces are only covered by the \
             rows that dispatch one"
        );
    }
}

#[test]
fn the_content_test_can_see_content_when_there_is_some() {
    let _serialised = serialised();
    // The positive control for `turn_span_records_no_content`. Without it that
    // test cannot tell "nothing leaked" from "nothing ran": each value below
    // is logged at DEBUG deliberately, so each must be visible when asked for.
    let ok = run(Level::TRACE, two_round_script(), ScriptedTools::ok());
    for (what, sentinel) in [
        ("a tool call's arguments", TOOL_ARGUMENT_SENTINEL),
        ("a tool's output", TOOL_RESULT_SENTINEL),
    ] {
        assert!(
            ok.console.contains(sentinel),
            "{what} belongs at DEBUG, so an operator who needs it can ask\n\
             --- console ---\n{}",
            ok.console
        );
    }

    let failed = run(Level::TRACE, two_round_script(), ScriptedTools::failing());
    assert!(
        failed.console.contains(TOOL_ERROR_SENTINEL),
        "a failing tool's message belongs at DEBUG too\n--- console ---\n{}",
        failed.console
    );
}

#[test]
fn every_probed_span_actually_opened() {
    let _serialised = serialised();
    // A content test says "nothing leaked". It cannot distinguish that from
    // "nothing ran". This asserts the spans the other tests read really exist,
    // so deleting the instrumentation fails by name here rather than turning
    // every leak test vacuously green.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let names = captured.span_names();

    for expected in ["turn", "turn.round", "llm.call", "tool.call"] {
        assert!(
            names.contains(&expected),
            "no `{expected}` span was opened; the run produced {names:?}"
        );
    }
}

/// A tool name shaped like a whole log line, which is what a model can put in
/// one. Ends with an ANSI escape, which survives with colour off.
const FORGED_TOOL_NAME: &str =
    "write_note\n2026-01-01T00:00:00.0Z ERROR forged: the database is on fire\u{1b}[31m";

#[test]
fn a_model_chosen_tool_name_cannot_forge_a_log_line() {
    let _serialised = serialised();
    // The tool name is the one field in the turn path the model writes, and it
    // reaches a span field, a metric label and a log line. Nothing else bounds
    // it: a newline produces what reads as a second genuine line, complete
    // with its own timestamp and level.
    let script = vec![
        LlmResponse::with_tool_calls(
            "",
            vec![ToolCall::new(
                "c1",
                FORGED_TOOL_NAME,
                serde_json::json!({}).to_string(),
            )],
        )
        .with_usage(usage(100, 10))
        .into(),
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(200, 20))
            .into(),
    ];
    let tools = ScriptedTools {
        tools: vec![ToolDefinition::new(
            FORGED_TOOL_NAME,
            "write a note",
            serde_json::json!({"type": "object"}),
        )],
        failure: None,
        cancel_after_first: false,
        unadvertised: Vec::new(),
        oversized_output: false,
        search_hit: None,
    };
    let captured = run(Level::INFO, script, tools);

    let tool = captured.span("tool.call");
    let rendered = tool.field("tool").expect("the tool span names its tool");
    assert!(
        !rendered.contains('\n') && !rendered.contains('\u{1b}'),
        "a model-chosen name must be sanitised before it reaches a span field; \
         got {rendered:?}"
    );
    assert!(
        rendered.starts_with("write_note"),
        "and the real name must still be legible; got {rendered:?}"
    );
    // The name itself still appears, and should: it is what the model asked
    // for and an operator needs to see it. What must not happen is a *line*
    // that the model wrote - with its own timestamp column, level and target.
    assert!(
        !captured
            .console
            .lines()
            .any(|l| l.starts_with("2026-01-01") || l.contains('\u{1b}')),
        "no console line may begin with a timestamp the model wrote, or carry an \
         escape it chose\n--- console ---\n{}",
        captured.console
    );
    let labels = label_values_in_window(&captured.after);
    assert!(
        !labels.iter().any(|v| v.contains('\n')),
        "and no metric label may carry a newline; got {labels:?}"
    );
}

#[test]
fn an_invented_tool_name_cannot_burn_the_metric_budget() {
    let _serialised = serialised();
    // The registry caps a metric at 64 distinct label sets, first come, with
    // no eviction. A tool name comes straight off the model's reply and an
    // invented one is dispatched, fails, and is recorded like any other - so
    // without a bound, sixty-four invented names, about sixty-four rounds of
    // one conversation, would kill per-tool latency until the process
    // restarts. Every real tool afterwards would fold into `cardinality=other`.
    let mut script: Vec<Reply> = (0..8)
        .map(|i| {
            LlmResponse::with_tool_calls(
                "",
                vec![ToolCall::new(
                    format!("c{i}"),
                    format!("invented_tool_{i}"),
                    serde_json::json!({}).to_string(),
                )],
            )
            .with_usage(usage(10, 1))
            .into()
        })
        .collect();
    script.push(
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(20, 2))
            .into(),
    );
    let captured = run(Level::INFO, script, ScriptedTools::ok());

    let values = label_values_in_window(&captured.after);
    let invented: Vec<&String> = values
        .iter()
        .filter(|v| v.starts_with("invented_tool_"))
        .collect();
    assert!(
        invented.is_empty(),
        "a name this turn never advertised must not become its own series; \
         got {invented:?}"
    );
    assert_eq!(
        captured.histogram_delta("tool.call.duration", &["tool=unknown"]),
        8,
        "the dispatches are still counted - under one bounded name, so an \
         operator can see that unadvertised names are being called at all"
    );
}

/// A fleet tool the executor knows, which this turn's round never offered.
const UNADVERTISED_TOOL: &str = "mcp_calendar_list_events";

#[test]
fn a_tool_the_daemon_knows_keeps_its_name_even_when_this_round_did_not_offer_it() {
    let _serialised = serialised();
    // `activated_tools` is per turn, so a fleet tool the model learned about in
    // an earlier turn and calls directly now is offered by nothing this round
    // - and still runs, because the executor's routing outlives the turn.
    // Judging on the offer alone files every one of those under `unknown` and
    // quietly empties that tool's latency series, which is the axis the bound
    // exists to protect.
    let script = vec![
        LlmResponse::with_tool_calls(
            "",
            vec![ToolCall::new(
                "c1",
                UNADVERTISED_TOOL,
                serde_json::json!({}).to_string(),
            )],
        )
        .with_usage(usage(100, 10))
        .into(),
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(200, 20))
            .into(),
    ];
    let captured = run(
        Level::INFO,
        script,
        ScriptedTools::knowing(UNADVERTISED_TOOL),
    );

    assert_eq!(
        captured.histogram_delta(
            "tool.call.duration",
            &[&format!("tool={UNADVERTISED_TOOL}")]
        ),
        1,
        "a tool the daemon's own list contains is recorded under its own name, \
         whether or not this round advertised it"
    );
    assert_eq!(
        captured.histogram_delta("tool.call.duration", &["tool=unknown"]),
        0,
    );
}

#[test]
fn an_advertised_tool_keeps_its_own_name() {
    let _serialised = serialised();
    // The other half of the bound. Folding every name to `unknown` would pass
    // the test above and destroy the axis it exists to protect.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    assert_eq!(
        captured.histogram_delta("tool.call.duration", &["tool=write_note"]),
        1,
        "a tool the turn advertised is recorded under its own name"
    );
}

// ---------------------------------------------------------------------------
// The profiler this replaces.
// ---------------------------------------------------------------------------

#[test]
fn profiling_llm_client_is_gone() {
    // `ProfilingLlmClient` was the only way to see inside a turn, and it paid
    // for that with a rotation-less JSONL file on the pod's ephemeral disk
    // carrying a 200-character preview of every message, with no conversation
    // or user id on any entry. Spans replace it with the correlation it never
    // had, and keeping both would mean two latency systems.
    let module = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("ports")
        .join("llm_profiling.rs");
    assert!(!module.exists(), "{} still exists", module.display());
}

// ---------------------------------------------------------------------------
// Carrying the trace across a boundary (epic D12 and D13).
// ---------------------------------------------------------------------------

/// The trace id the correlation id spells, with the hyphens a uuid carries and
/// a trace id does not.
fn trace_id_of(request_id: &str) -> String {
    request_id.replace('-', "")
}

#[test]
fn request_id_and_trace_id_are_the_same_value() {
    // The whole point of D12: one identifier in the client's event stream, in
    // the pod log and in the backend, pasteable from any one into any other.
    // A uuid is 16 bytes and a W3C trace id is 16 bytes, so no mapping table
    // and no second identifier exist to disagree.
    //
    // This holds with the `otel` feature off, which is the build every desktop
    // install runs. What the feature adds is export, not correlation.
    //
    // It holds for a turn whose trace was minted here, which is every turn
    // except one forwarded by a caller that already had a trace. That case is
    // deliberate and is pinned by
    // `incoming_traceparent_is_continued_not_replaced`, which asserts the two
    // differ.
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    assert_eq!(
        turn.field("request_id"),
        Some(REQUEST_ID),
        "the turn span must carry the id the client already correlates by"
    );
    assert_eq!(
        turn.field("trace_id"),
        Some(trace_id_of(REQUEST_ID).as_str()),
        "the trace id must be the request id and nothing else; got {:?}",
        turn.fields
    );
}

#[test]
fn incoming_traceparent_is_continued_not_replaced() {
    // A caller that already has a trace is joined, not restarted. Without
    // this the web BFF's hop becomes a second trace that only a timestamp
    // relates to the first.
    let _serialised = serialised();
    let incoming = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let captured = capture(Level::INFO, async move {
        let handler = handler(two_round_script(), ScriptedTools::ok());
        one_turn_traced(
            &handler,
            resolve_turn_trace(Some(incoming), REQUEST_ID, CONVERSATION_ID),
        )
        .await;
    });

    let turn = captured.span("turn");
    assert_eq!(
        turn.field("trace_id"),
        Some("4bf92f3577b34da6a3ce929d0e0e4736"),
        "the turn must join the caller's trace"
    );
    assert_eq!(
        turn.field("request_id"),
        Some(REQUEST_ID),
        "continuing a trace must not disturb the correlation id, which is a \
         separate identifier the client reads its own stream by"
    );
}

#[test]
fn conversation_id_is_an_attribute_not_a_trace() {
    // D13. A conversation lives for days and holds an unbounded number of
    // turns, which no backend renders usefully. So two turns in one
    // conversation are two traces that share an attribute, and one query still
    // returns every turn in the conversation.
    let _serialised = serialised();
    let first = "11111111-2222-4333-8444-555555555555";
    let second = "99999999-8888-4777-8666-555555555555";

    let mut trace_ids = Vec::new();
    for request_id in [first, second] {
        let captured = capture(Level::INFO, async move {
            let handler = handler(two_round_script(), ScriptedTools::ok());
            one_turn_traced(
                &handler,
                resolve_turn_trace(None, request_id, CONVERSATION_ID),
            )
            .await;
        });
        let turn = captured.span("turn");
        assert_eq!(
            turn.field("conversation_id"),
            Some(CONVERSATION_ID),
            "both turns must carry the conversation as an attribute"
        );
        trace_ids.push(
            turn.field("trace_id")
                .expect("every turn span carries a trace id")
                .to_string(),
        );
    }

    assert_eq!(trace_ids[0], trace_id_of(first));
    assert_eq!(trace_ids[1], trace_id_of(second));
    assert_ne!(
        trace_ids[0], trace_ids[1],
        "two turns in one conversation must be two traces, or the trace grows \
         without bound and no backend can draw it"
    );
}

#[test]
fn every_span_in_a_turn_carries_the_conversation_id() {
    // The attribute is only useful if the query that reads it returns the
    // whole turn. A round or a provider call without it drops out of that
    // answer.
    let _serialised = serialised();
    let captured = capture(Level::INFO, async move {
        let handler = handler(two_round_script(), ScriptedTools::ok());
        one_turn_traced(
            &handler,
            resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID),
        )
        .await;
    });

    for name in ["turn", "turn.round", "llm.call", "tool.call"] {
        let spans = captured.spans_named(name);
        assert!(!spans.is_empty(), "no `{name}` span opened");
        for span in spans {
            assert_eq!(
                span.field("conversation_id"),
                Some(CONVERSATION_ID),
                "a `{name}` span must carry the conversation id; got {:?}",
                span.fields
            );
        }
    }
}

#[test]
fn llm_span_records_the_provider_request_id() {
    // No provider continues our trace, so the useful move at that boundary is
    // capture: record the identifier a support ticket quotes. It lands on the
    // provider-call span because that is the span open while the call runs.
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let calls = captured.spans_named("llm.call");
    assert!(
        !calls.is_empty(),
        "no `llm.call` span opened, so this test would pass without any \
         provider call happening at all; the run produced {:?}",
        captured.span_names()
    );
    for span in calls {
        assert_eq!(
            span.field("provider_request_id"),
            Some(PROVIDER_REQUEST_ID),
            "a provider call must record what the provider called it; got {:?}",
            span.fields
        );
    }
}

// ---------------------------------------------------------------------------
// Token counts on the provider-call span.
//
// The counts are already metrics, labelled by provider and model. That answers
// "how many tokens did this model burn today" and cannot answer "what did this
// turn cost", because a conversation id is unbounded and the registry caps a
// metric at 64 label sets with no eviction. A span attribute has no such
// budget, and the provider-call span already carries the conversation id and
// the round - so the numbers belong there too.
//
// Every assertion below reads the span back in process. A span fixes its field
// set when it opens, so a count that arrives afterwards has to be recorded into
// a field declared empty at creation; `record` on a field the span never
// declared is dropped with no warning, and the span exports without it. Nothing
// prints a span field either, so a console assertion cannot see the difference.
// ---------------------------------------------------------------------------

/// The provider call one round made, found by parentage rather than by
/// position: a turn also makes provider calls outside its rounds.
fn round_llm_call(captured: &Captured, round: usize) -> &SpanRecord {
    let rounds = captured.spans_named("turn.round");
    let parent = rounds
        .get(round)
        .unwrap_or_else(|| {
            panic!(
                "the turn ran fewer than {} rounds, so there is no call to read; \
                 the run produced {:?}",
                round + 1,
                captured.span_names()
            )
        })
        .id
        .clone();
    captured
        .spans_named("llm.call")
        .into_iter()
        .find(|s| s.parent.as_ref() == Some(&parent))
        .unwrap_or_else(|| {
            panic!(
                "round {round} opened no `llm.call` span, so this test would \
                 pass without any provider call happening at all; the run \
                 produced {:?}",
                captured.span_names()
            )
        })
}

#[test]
fn llm_call_span_carries_every_token_count_a_caching_provider_reports() {
    let _serialised = serialised();
    let script = vec![
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(caching_usage(100, 10, 25, 50))
            .into(),
    ];
    let captured = run(Level::INFO, script, ScriptedTools::ok());
    let call = round_llm_call(&captured, 0);

    assert_eq!(
        call.field(GEN_AI_INPUT_TOKENS),
        Some("100"),
        "got {:?}",
        call.fields
    );
    assert_eq!(
        call.field(GEN_AI_OUTPUT_TOKENS),
        Some("10"),
        "got {:?}",
        call.fields
    );
    assert_eq!(
        call.field(GEN_AI_CACHE_CREATION_INPUT_TOKENS),
        Some("25"),
        "a cache write costs real money and is invisible in the input count \
         alone; got {:?}",
        call.fields
    );
    assert_eq!(
        call.field(GEN_AI_CACHE_READ_INPUT_TOKENS),
        Some("50"),
        "a cache read costs a fraction of a fresh input token, so a well-cached \
         call reads as a cold one without it; got {:?}",
        call.fields
    );
}

#[test]
fn llm_call_span_leaves_the_cache_counts_off_a_provider_that_does_not_cache() {
    let _serialised = serialised();
    // Most providers report two counts and nothing about a cache, because they
    // have none. Their calls must carry the two they do report, and must not
    // claim a cache write of zero.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let call = round_llm_call(&captured, 0);

    assert_eq!(
        call.field(GEN_AI_INPUT_TOKENS),
        Some("100"),
        "got {:?}",
        call.fields
    );
    assert_eq!(
        call.field(GEN_AI_OUTPUT_TOKENS),
        Some("10"),
        "got {:?}",
        call.fields
    );
    for absent in [
        GEN_AI_CACHE_CREATION_INPUT_TOKENS,
        GEN_AI_CACHE_READ_INPUT_TOKENS,
    ] {
        assert_eq!(
            call.field(absent),
            None,
            "`{absent}` must be absent, not zero: this provider does no prompt \
             caching, and a recorded `0` reads as a real measurement; got {:?}",
            call.fields
        );
    }
}

#[test]
fn llm_call_span_records_no_token_count_when_the_provider_reports_none() {
    let _serialised = serialised();
    // A connector that reports no usage at all. `llm.tokens.unreported` counts
    // exactly this case, and a zero on the span would undermine it: an
    // undercount that looks like data is worse than a gap that looks like one.
    let script = vec![LlmResponse::text(REPLY_SENTINEL).into()];
    let captured = run(Level::INFO, script, ScriptedTools::ok());
    let call = round_llm_call(&captured, 0);

    for absent in GEN_AI_TOKEN_ATTRIBUTES {
        assert_eq!(
            call.field(absent),
            None,
            "the provider reported nothing, so `{absent}` must carry nothing - \
             not a zero, which sums into a total that looks real; got {:?}",
            call.fields
        );
    }
}

#[test]
fn a_failed_provider_call_closes_its_span_and_carries_no_token_count() {
    let _serialised = serialised();
    // An error path has no usage to report and must still close its span, so a
    // trace of a failed turn draws the call at the length it really took.
    let script = vec![Reply::Fail(CoreError::Llm("provider unavailable".into()))];
    let captured = run(Level::INFO, script, ScriptedTools::ok());
    let call = round_llm_call(&captured, 0);

    assert_eq!(
        call.field("outcome"),
        Some("error"),
        "this test only means something if the call really failed; got {:?}",
        call.fields
    );
    assert!(
        call.closed_at.is_some(),
        "a failed provider call must still close its span, or the trace draws \
         it running to the end of the turn"
    );
    for absent in GEN_AI_TOKEN_ATTRIBUTES {
        assert_eq!(
            call.field(absent),
            None,
            "a call that failed reported no usage, so `{absent}` must be \
             absent; got {:?}",
            call.fields
        );
    }
}

#[test]
fn a_provider_call_outside_a_round_carries_its_token_counts_too() {
    let _serialised = serialised();
    // A turn spends provider time outside its rounds - here the title for a new
    // conversation. Those calls cost tokens like any other, and they open their
    // own `llm.call` span, so the same numbers have to reach it.
    let script = vec![
        LlmResponse::text(REPLY_SENTINEL)
            .with_usage(usage(100, 10))
            .into(),
        LlmResponse::text("Example Channel Name")
            .with_usage(caching_usage(7, 3, 1, 2))
            .into(),
    ];
    let captured = run(Level::INFO, script, ScriptedTools::ok());

    let title = captured
        .spans_named("llm.call")
        .into_iter()
        .find(|s| s.field("purpose") == Some("title"))
        .unwrap_or_else(|| {
            panic!(
                "the turn's first message must generate a title through the \
                 provider, or this test proves nothing; the run produced {:?}",
                captured.span_names()
            )
        });

    assert_eq!(
        title.field(GEN_AI_INPUT_TOKENS),
        Some("7"),
        "got {:?}",
        title.fields
    );
    assert_eq!(
        title.field(GEN_AI_OUTPUT_TOKENS),
        Some("3"),
        "got {:?}",
        title.fields
    );
    assert_eq!(
        title.field(GEN_AI_CACHE_CREATION_INPUT_TOKENS),
        Some("1"),
        "got {:?}",
        title.fields
    );
    assert_eq!(
        title.field(GEN_AI_CACHE_READ_INPUT_TOKENS),
        Some("2"),
        "got {:?}",
        title.fields
    );
}

#[test]
fn propagation_records_no_content() {
    // D10 on the fields this change adds. A span records its fields at
    // creation and nothing prints them, so a captured value is invisible on
    // the console and still exports when the span closes. These are read back
    // in process for that reason.
    let _serialised = serialised();
    let captured = capture(Level::INFO, async move {
        let handler = handler(two_round_script(), ScriptedTools::failing());
        one_turn_traced(
            &handler,
            resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID),
        )
        .await;
    });

    let added = ["trace_id", "conversation_id", "provider_request_id"];
    let content = [
        ("prompt", PROMPT_SENTINEL),
        ("tool argument", TOOL_ARGUMENT_SENTINEL),
        ("model reply", REPLY_SENTINEL),
        ("tool result", TOOL_RESULT_SENTINEL),
        ("tool error", TOOL_ERROR_SENTINEL),
    ];

    for span in &captured.spans {
        for field in added {
            let Some(value) = span.field(field) else {
                continue;
            };
            for (what, sentinel) in content {
                assert!(
                    !value.contains(sentinel),
                    "the {what} reached `{}.{field}`, which exports verbatim \
                     when the span closes",
                    span.name
                );
            }
        }
    }
}

/// A provider request id shaped like a whole log line. A response header is
/// bounded by nothing at this end: an upstream proxy, or a provider having a
/// bad day, can answer with anything at all.
const FORGED_PROVIDER_REQUEST_ID: &str =
    "req_1\n2026-01-01T00:00:00.0Z ERROR forged: the database is on fire\u{1b}[31m";

#[test]
fn a_provider_request_id_cannot_forge_a_log_line() {
    // The value comes from a host nobody here controls and lands on a span
    // field, which exports verbatim when the span closes. A newline produces
    // what reads as a second genuine line, complete with its own timestamp and
    // level, and an ANSI escape survives with colour off.
    let _serialised = serialised();
    let captured = capture(Level::INFO, async move {
        let handler = ConversationHandler::with_tools(
            MemStore::default(),
            ScriptedLlm::reporting(two_round_script(), FORGED_PROVIDER_REQUEST_ID),
            ScriptedTools::ok(),
            Box::new(|| CONVERSATION_ID.to_string()),
        );
        one_turn(&handler).await;
    });

    let calls = captured.spans_named("llm.call");
    assert!(
        !calls.is_empty(),
        "no `llm.call` span opened, so the field assertions below would be \
         vacuous"
    );
    for span in calls {
        let Some(recorded) = span.field("provider_request_id") else {
            continue;
        };
        assert!(
            !recorded.contains('\n') && !recorded.contains('\u{1b}'),
            "a provider's own header must be neutralised before it reaches a \
             span field; got {recorded:?}"
        );
    }

    assert!(
        !captured
            .console
            .contains("forged: the database is on fire\u{1b}"),
        "the escape survived onto the console\n--- console ---\n{}",
        captured.console
    );
    for line in captured.console.lines() {
        assert!(
            !line
                .trim_start()
                .starts_with("2026-01-01T00:00:00.0Z ERROR forged"),
            "a provider header forged a line that reads as the daemon's own\n\
             --- console ---\n{}",
            captured.console
        );
    }
}

/// A conversation id shaped like a whole log line. The id a turn carries is
/// the one that arrived on the wire, and nothing between the socket and the
/// span bounds it.
const FORGED_CONVERSATION_ID: &str =
    "c-91\n2026-01-01T00:00:00.0Z ERROR forged: the database is on fire\u{1b}[31m";

#[test]
fn a_conversation_id_cannot_forge_a_log_line() {
    // Every span in a turn now carries the conversation id, and a `%` field
    // reaches the console through `Display`, which does not escape what
    // `Debug` would. The daemon mints conversation ids itself today, so this
    // is defence in depth - but that invariant lives in another function, is
    // not asserted anywhere, and would not survive a store that adopts a
    // client-supplied id.
    let _serialised = serialised();
    let captured = capture(Level::INFO, async move {
        let handler = ConversationHandler::with_tools(
            MemStore::default(),
            ScriptedLlm::new(two_round_script()),
            ScriptedTools::ok(),
            Box::new(|| FORGED_CONVERSATION_ID.to_string()),
        );
        one_turn(&handler).await;
    });

    let mut carried = 0;
    for span in &captured.spans {
        let Some(recorded) = span.field("conversation_id") else {
            continue;
        };
        carried += 1;
        assert!(
            !recorded.contains('\u{1b}'),
            "an escape survived onto `{}.conversation_id`, which exports \
             verbatim when the span closes; got {recorded:?}",
            span.name
        );
    }

    assert!(
        carried > 0,
        "no span carried a conversation id, so the loop above asserted nothing; \
         the run produced {:?}",
        captured.span_names()
    );

    for line in captured.console.lines() {
        assert!(
            !line
                .trim_start()
                .starts_with("2026-01-01T00:00:00.0Z ERROR forged"),
            "a conversation id forged a line that reads as the daemon's own\n\
             --- console ---\n{}",
            captured.console
        );
    }
}

// ---------------------------------------------------------------------------
// What filled the input (#1203).
//
// `llm.tokens.input` says a round cost 40k and cannot say whether that was the
// transcript, the pinned notes or eighty tool schemas, and each of those has a
// different fix. These tests are the promise that the breakdown is separable
// in the way an operator would act on it.
//
// Every name below is spelled out rather than imported from the crate under
// test, so a rename at the recording site fails these tests instead of
// travelling with them.
// ---------------------------------------------------------------------------

/// The turn-span field each part of an assembled prompt is reported under.
const PROMPT_PART_FIELDS: [&str; 10] = [
    "prompt.system_tokens",
    "prompt.summary_tokens",
    "prompt.current_task_tokens",
    "prompt.working_state_tokens",
    "prompt.plan_tokens",
    "prompt.pinned_tokens",
    "prompt.scratchpad_tokens",
    "prompt.recall_tokens",
    "prompt.transcript_tokens",
    "prompt.tool_schema_tokens",
];

/// What the parts add up to.
const PROMPT_TOTAL_FIELD: &str = "prompt.total_tokens";

/// How many tool schemas the round advertised. A count, not a token figure,
/// and named so (#1212).
const PROMPT_TOOL_COUNT_FIELD: &str = "prompt.tool_count";

/// Every field the breakdown puts on the turn span.
fn prompt_fields() -> Vec<&'static str> {
    let mut all = PROMPT_PART_FIELDS.to_vec();
    all.push(PROMPT_TOTAL_FIELD);
    all.push(PROMPT_TOOL_COUNT_FIELD);
    all.push(PROMPT_TOOL_TOKENS_PEAK_FIELD);
    all.push(PROMPT_TOOL_COUNT_PEAK_FIELD);
    all
}

const PROMPT_PART_TOKENS_METRIC: &str = "llm.prompt.part.tokens";
const PROMPT_TOOLS_METRIC: &str = "llm.prompt.tools";
const PROMPT_MEASURED_METRIC: &str = "llm.prompt.measured";

/// The `part` label values [`PROMPT_PART_TOKENS_METRIC`] may carry - the whole
/// bounded set.
const PROMPT_PART_LABELS: [&str; 10] = [
    "system",
    "summary",
    "current_task",
    "working_state",
    "plan",
    "pinned",
    "scratchpad",
    "recall",
    "transcript",
    "tool_schemas",
];

/// Read one token figure off the turn span, failing by name when it is absent.
fn prompt_field(span: &SpanRecord, field: &str) -> u64 {
    let raw = span.field(field).unwrap_or_else(|| {
        panic!(
            "the turn span carries no `{field}`; it recorded {:?}",
            span.fields
        )
    });
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("`{field}` must be a number; got {raw:?} ({e})"))
}

/// A pinned note long enough that its block costs real tokens.
const PINNED_NOTE: &str = "the registry caps a metric at sixty-four label sets, \
     first come, with no eviction, so an unbounded label is an unbounded leak \
     in a process that runs for weeks and the dimension stays dead until the \
     process restarts";

/// A handler whose scratchpad holds one pinned note, so the turn assembles a
/// `[Pinned]` block. Nothing else is wired, so `[Scratchpad]` stays gated
/// silent and `[Recall]` never renders.
fn handler_with_pinned_note(
    responses: Vec<Reply>,
    tools: ScriptedTools,
) -> ConversationHandler<MemStore, ScriptedLlm, ScriptedTools> {
    use desktop_assistant_core::domain::ScratchpadNote;
    use desktop_assistant_core::ports::scratchpad::ScratchpadListFn;
    let list: ScratchpadListFn = Arc::new(move |conversation_id: String, _note_type, _limit| {
        let mut note = ScratchpadNote::new("sp-1", conversation_id, "cap", PINNED_NOTE);
        note.pinned = true;
        Box::pin(async move { Ok(vec![note]) })
    });
    handler(responses, tools).with_scratchpad_list(list)
}

// ---------------------------------------------------------------------------
// What the sweep did not reach (#1205).
//
// Names are spelled out rather than imported, for the reason the block above
// gives: a rename at the recording site must fail these tests instead of
// travelling with them.
// ---------------------------------------------------------------------------

/// The turn-span fields the tool-byte census reports under.
const CONTEXT_TOOL_BYTE_FIELDS: [&str; 6] = [
    "context.tool_bytes",
    "context.tool_bytes_evicted",
    "context.tool_bytes_reduced",
    "context.tool_bytes_shrunk_elsewhere",
    "context.tool_bytes_carried",
    "context.tool_carried_pct",
];

/// The fraction the epic is measured against. Deliberately without `_bytes` in
/// its name: a unit check by substring passes anything carrying that word,
/// whatever the value actually is, and this is the one field here that is not
/// a byte count.
const CONTEXT_CARRIED_PCT_FIELD: &str = "context.tool_carried_pct";

#[test]
fn turn_span_reports_what_eviction_did_not_reach() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    for field in CONTEXT_TOOL_BYTE_FIELDS {
        assert!(
            turn.field(field).is_some(),
            "the turn span must carry `{field}`; it recorded {:?}",
            turn.fields
        );
    }
    assert!(
        prompt_field(turn, "context.tool_bytes") > 0,
        "a turn that ran a tool held tool bytes; it recorded {:?}",
        turn.fields
    );
    assert!(
        !CONTEXT_CARRIED_PCT_FIELD.contains("_bytes"),
        "the one field here that is not a byte count must not be named as if \
         it were, or the unit check below passes it on the substring alone"
    );
    // Per model needs no label: the turn span already carries the route.
    assert!(
        turn.field("model").is_some(),
        "the census is read per model off the turn span; it recorded {:?}",
        turn.fields
    );
}

/// The census must cover the WINDOW, not the conversation. A conversation
/// carries every tool result it ever held and the store loads all of them;
/// counting those would make the figure track how old a conversation is rather
/// than what the turn is carrying.
///
/// The fixture makes the two answers differ by an order of magnitude: one
/// ancient result far outside the window, one small one inside it.
#[test]
fn the_census_covers_the_window_and_not_the_whole_conversation() {
    let _serialised = serialised();

    /// Bytes of the ancient result. Far larger than anything the turn itself
    /// produces, so a census over the conversation cannot be mistaken for one
    /// over the window.
    const ANCIENT_BYTES: usize = 400_000;

    let captured = capture(Level::INFO, async move {
        // Seed a conversation whose OLDEST turn holds a huge tool result, then
        // bury it under enough turns that the window cannot reach it.
        let mut seeded = Conversation::new(CONVERSATION_ID, "c");
        seeded
            .messages
            .push(Message::new(Role::User, "the oldest ask"));
        seeded
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "ancient",
                "read_file",
                "{}",
            )]));
        seeded
            .messages
            .push(Message::tool_result("ancient", "x".repeat(ANCIENT_BYTES)));
        for i in 0..60 {
            seeded
                .messages
                .push(Message::new(Role::User, format!("u-{i}")));
            seeded
                .messages
                .push(Message::new(Role::Assistant, format!("a-{i}")));
        }
        let store = MemStore::default();
        store.create(seeded).await.expect("seed the conversation");

        // A windowing conversation compacts on entry, and the summariser is
        // the same scripted client - so the script carries several tool rounds
        // to be sure one survives being eaten and the loop reaches the census
        // at the bottom of a round.
        let mut script: Vec<Reply> = (0..4)
            .map(|_| {
                LlmResponse::with_tool_calls("", vec![tool_call("c1")])
                    .with_usage(usage(100, 10))
                    .into()
            })
            .collect();
        script.push(
            LlmResponse::text(REPLY_SENTINEL)
                .with_usage(usage(200, 20))
                .into(),
        );

        let handler = ConversationHandler::with_tools(
            store,
            ScriptedLlm::new(script),
            ScriptedTools::ok(),
            Box::new(|| CONVERSATION_ID.to_string()),
        );
        let conv = ConversationId::from(CONVERSATION_ID);

        with_user_id(
            UserId::new(USER_ID),
            with_request_id(
                REQUEST_ID.to_string(),
                with_turn_route(route(), async {
                    let _ = handler
                        .send_prompt_with_override(
                            &conv,
                            PROMPT_SENTINEL.to_string(),
                            None,
                            String::new(),
                            Box::new(|_| true),
                            Box::new(|_| {}),
                            CancellationToken::new(),
                        )
                        .await;
                }),
            ),
        )
        .await;
    });

    let turn = captured.span("turn");
    let total = prompt_field(turn, "context.tool_bytes");
    assert!(
        total > 0,
        "precondition: the turn ran a tool, so it held some tool bytes"
    );
    assert!(
        total < ANCIENT_BYTES as u64 / 10,
        "the census counted a result the window dropped: {total} bytes against \
         an ancient result of {ANCIENT_BYTES}"
    );
}

#[test]
fn every_context_figure_on_the_turn_span_names_its_unit() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    let mut checked = 0;
    for key in turn.fields.keys() {
        if !key.starts_with("context.") {
            continue;
        }
        checked += 1;
        assert!(
            key.contains("_bytes") || key == CONTEXT_CARRIED_PCT_FIELD,
            "`{key}` states no unit. A byte count and a percentage look \
             equally plausible side by side, so every figure here says which \
             it is"
        );
    }
    assert_eq!(
        checked,
        CONTEXT_TOOL_BYTE_FIELDS.len(),
        "the turn span carried {checked} `context.` fields, so this test did \
         not see the set it names; it recorded {:?}",
        turn.fields
    );
}

#[test]
fn turn_span_carries_a_token_figure_for_every_prompt_part() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    for field in prompt_fields() {
        assert!(
            turn.field(field).is_some(),
            "the turn span must carry `{field}`; it recorded {:?}",
            turn.fields
        );
    }
    assert!(
        prompt_field(turn, PROMPT_TOTAL_FIELD) > 0,
        "a turn that assembled a prompt must report a non-zero total; it \
         recorded {:?}",
        turn.fields
    );
}

#[test]
fn every_prompt_figure_on_the_turn_span_names_its_unit() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    let mut checked = 0;
    for key in turn.fields.keys() {
        let Some(rest) = key.strip_prefix("prompt.") else {
            continue;
        };
        checked += 1;
        assert!(
            rest.contains("_tokens")
                || key == PROMPT_TOOL_COUNT_FIELD
                || key == PROMPT_TOOL_COUNT_PEAK_FIELD,
            "`{key}` states no unit. A character count and a token count for \
             the same block look equally plausible side by side, so every \
             figure here says which it is"
        );
    }
    assert_eq!(
        checked,
        prompt_fields().len(),
        "the turn span carried {checked} `prompt.` fields, so this test did \
         not see the set it names; it recorded {:?}",
        turn.fields
    );
}

#[test]
fn prompt_parts_are_recorded_when_the_provider_reports_no_token_counts() {
    let _serialised = serialised();
    // A connector that reports no usage at all: what the breakdown must not
    // depend on, because the parts are measured here and not by the provider.
    let captured = run(
        Level::INFO,
        vec![LlmResponse::text(REPLY_SENTINEL).into()],
        ScriptedTools::ok(),
    );
    assert!(
        captured.console.contains("input_tokens=-"),
        "precondition: the provider must have reported nothing\n\
         --- console ---\n{}",
        captured.console
    );

    let turn = captured.span("turn");
    for field in prompt_fields() {
        assert!(
            turn.field(field).is_some(),
            "`{field}` must be recorded whatever the provider reported; the \
             turn recorded {:?}",
            turn.fields
        );
    }
    assert!(prompt_field(turn, PROMPT_TOTAL_FIELD) > 0);
}

#[test]
fn a_turn_with_no_pinned_notes_recall_or_scratchpad_records_zero_for_those_parts() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    for field in [
        "prompt.pinned_tokens",
        "prompt.recall_tokens",
        "prompt.scratchpad_tokens",
    ] {
        assert_eq!(
            turn.field(field),
            Some("0"),
            "a block that did not render is a measured zero, not an omission - \
             an absent field cannot be told from a part nobody measured; the \
             turn recorded {:?}",
            turn.fields
        );
    }
}

#[test]
fn a_pinned_note_is_counted_against_the_pinned_part_and_no_other() {
    let _serialised = serialised();
    let with_pin = capture(Level::INFO, async {
        let handler = handler_with_pinned_note(two_round_script(), ScriptedTools::ok());
        one_turn(&handler).await;
    });
    let without_pin = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let pinned = with_pin.span("turn");
    let bare = without_pin.span("turn");

    assert!(
        prompt_field(pinned, "prompt.pinned_tokens") > 0,
        "the pinned note must show up in the pinned part; the turn recorded \
         {:?}",
        pinned.fields
    );
    assert_eq!(
        prompt_field(bare, "prompt.pinned_tokens"),
        0,
        "precondition: the same turn without a pin must report no pinned cost"
    );
    for field in ["prompt.scratchpad_tokens", "prompt.recall_tokens"] {
        assert_eq!(
            prompt_field(pinned, field),
            0,
            "a pin is not a scratchpad index and not a recall offer, so it \
             must not be counted as `{field}`; the turn recorded {:?}",
            pinned.fields
        );
    }
    assert_eq!(
        prompt_field(pinned, "prompt.transcript_tokens"),
        prompt_field(bare, "prompt.transcript_tokens"),
        "the two turns sent the same messages, so a pin that changed the \
         transcript figure is being counted in the wrong part"
    );
    assert_eq!(
        prompt_field(pinned, "prompt.system_tokens"),
        prompt_field(bare, "prompt.system_tokens"),
        "the same reason: a pin is not part of the system instruction"
    );
}

#[test]
fn a_turn_cancelled_before_it_assembles_a_prompt_carries_no_prompt_field() {
    let _serialised = serialised();
    // The counterpart to the zero above, and what makes it mean anything: a
    // zero says a block did not render, and an absent field says no prompt was
    // ever built. A turn whose token is already tripped leaves at the loop's
    // first checkpoint, before assembly.
    let captured = capture(Level::INFO, async {
        let handler = handler(two_round_script(), ScriptedTools::ok());
        let conv = handler
            .create_conversation("c".into(), vec![])
            .await
            .expect("create the conversation");
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        with_user_id(
            UserId::new(USER_ID),
            with_request_id(
                REQUEST_ID.to_string(),
                with_turn_route(route(), async {
                    let _ = handler
                        .send_prompt_with_override(
                            &conv.id,
                            PROMPT_SENTINEL.to_string(),
                            None,
                            String::new(),
                            Box::new(|_| true),
                            Box::new(|_| {}),
                            cancelled,
                        )
                        .await;
                }),
            ),
        )
        .await;
    });

    let turn = captured.span("turn");
    assert_eq!(
        turn.field("outcome"),
        Some("cancelled"),
        "precondition: the turn must have left before it assembled anything; \
         it recorded {:?}",
        turn.fields
    );
    for field in prompt_fields() {
        assert_eq!(
            turn.field(field),
            None,
            "`{field}` must be absent when no prompt was assembled - a zero \
             here would read as a block that rendered nothing; the turn \
             recorded {:?}",
            turn.fields
        );
    }
}

#[test]
fn the_turn_span_carries_the_tool_count_and_the_tool_schema_cost() {
    let _serialised = serialised();
    // #1212 measured a turn that had spent 23.7k tokens on 99 tool schemas
    // before it did anything, and nobody could see it without reading a log by
    // hand. This is the pair that makes the tool set visible.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());
    let turn = captured.span("turn");

    assert_eq!(
        turn.field(PROMPT_TOOL_COUNT_FIELD),
        Some("1"),
        "the turn advertised exactly one tool; it recorded {:?}",
        turn.fields
    );
    assert!(
        prompt_field(turn, "prompt.tool_schema_tokens") > 0,
        "an advertised schema costs prompt tokens the message bodies never \
         show; the turn recorded {:?}",
        turn.fields
    );
}

#[test]
fn the_turn_span_reports_the_prompt_the_turn_opened_with_not_the_one_it_ended_with() {
    let _serialised = serialised();
    // A turn's rounds grow the transcript with their own tool traffic. The
    // figure here is the standing cost the turn entered with, so a later
    // round's multi-megabyte tool result must not become the transcript
    // number an operator reads for this turn.
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::oversized());
    let turn = captured.span("turn");

    let transcript = prompt_field(turn, "prompt.transcript_tokens");
    assert!(transcript > 0, "the user prompt is in the transcript");
    assert!(
        transcript < 1_000,
        "the turn opened with one short user prompt, so its transcript figure \
         must be small; a later round's capped tool result is tens of \
         thousands of tokens, and reading {transcript} here means the last \
         round overwrote the turn's own number"
    );
}

#[test]
fn prompt_part_metrics_are_labelled_by_part_alone() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    // Every series this metric holds, not only the ones that moved: a part
    // that rendered nothing records a zero, and a zero delta reads the same as
    // no measurement at all. What the series set can still say is which
    // label values exist, which is the bounded-cardinality claim.
    let mut seen_parts: Vec<String> = Vec::new();
    for counter in captured.after.counters.iter() {
        match counter.name {
            PROMPT_PART_TOKENS_METRIC => {
                let keys: Vec<&str> = counter.labels.iter().map(|l| l.key()).collect();
                assert_eq!(
                    keys,
                    vec!["part"],
                    "`{PROMPT_PART_TOKENS_METRIC}` carries the part name and \
                     nothing else - a conversation or model axis here would \
                     burn the registry's 64-label-set cap, which has no \
                     eviction"
                );
                let value = counter.labels[0].value().to_string();
                assert!(
                    PROMPT_PART_LABELS.contains(&value.as_str()),
                    "`{value}` is not one of the bounded part names \
                     {PROMPT_PART_LABELS:?}"
                );
                seen_parts.push(value);
            }
            PROMPT_TOOLS_METRIC | PROMPT_MEASURED_METRIC => {
                assert!(
                    counter.labels.is_empty(),
                    "`{}` takes no label at all; it carried {:?}",
                    counter.name,
                    counter.labels
                );
            }
            _ => {}
        }
    }

    seen_parts.sort();
    let mut expected: Vec<String> = PROMPT_PART_LABELS.iter().map(|p| p.to_string()).collect();
    expected.sort();
    assert_eq!(
        seen_parts, expected,
        "the metric holds one series per part and no others - the bound is \
         the whole point, and no turn in this file renders a summary, a \
         current-task anchor or a plan, so those three series exist only if \
         every part reports on every turn"
    );
    assert_eq!(
        captured.counter_delta(PROMPT_MEASURED_METRIC, &[]),
        1,
        "one turn measures one prompt, which is the denominator the per-part \
         totals are read against"
    );
    assert_eq!(
        captured.counter_delta(PROMPT_TOOLS_METRIC, &[]),
        1,
        "the turn advertised one tool"
    );
}

#[test]
fn no_prompt_metric_carries_a_conversation_scoped_label() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let mut checked = 0;
    for counter in captured.after.counters.iter().filter(|c| {
        matches!(
            c.name,
            PROMPT_PART_TOKENS_METRIC
                | PROMPT_TOOLS_METRIC
                | PROMPT_MEASURED_METRIC
                | PROMPT_TOOL_SERVER_TOKENS_METRIC
                | PROMPT_ROUND_TOOLS_METRIC
                | PROMPT_ROUND_MEASURED_METRIC
        )
    }) {
        checked += 1;
        for label in &counter.labels {
            // The key, first. A conversation axis added here would not
            // necessarily carry a recognisable value - the id may be unset by
            // the time the turn reports - and the axis is the leak whatever
            // the value turns out to be.
            assert!(
                // `part` is the closed set of ten prompt parts; `server` is
                // the connection a schema was offered by, bounded by the
                // operator's own server list (#1212). Nothing else.
                matches!(label.key(), "part" | "server"),
                "`{}` may carry no axis but the part or the server; a \
                 per-conversation, per-user or per-model axis is an unbounded \
                 series key in a registry that caps a metric at 64 label sets \
                 and evicts none. It carried {:?}",
                counter.name,
                counter.labels
            );
            for forbidden in [CONVERSATION_ID, USER_ID, REQUEST_ID, MODEL, PROVIDER] {
                assert_ne!(
                    label.value(),
                    forbidden,
                    "`{}` must not be labelled by conversation, user, request \
                     or model; it carried {:?}",
                    counter.name,
                    counter.labels
                );
            }
        }
    }
    assert!(
        checked > 0,
        "no prompt metric was recorded, so this test asserted nothing"
    );
}

#[test]
fn the_docs_state_the_parts_are_estimates_that_do_not_sum_to_the_provider_count() {
    // The parts come from this daemon's own estimator and the provider counts
    // its own way, so the two disagree by construction. Said once, in the
    // operator-facing document, rather than per field.
    const LOGGING_DOC: &str = include_str!("../../../docs/logging.md");
    // Read with runs of whitespace collapsed, so a claim the document happens
    // to wrap across two lines still counts as made.
    let doc: String = LOGGING_DOC.split_whitespace().collect::<Vec<_>>().join(" ");

    for name in prompt_fields() {
        assert!(
            doc.contains(name),
            "`{name}` is recorded and undocumented, so nobody reading \
             docs/logging.md knows it exists"
        );
    }
    for name in [ROUND_TOOL_COUNT_FIELD, ROUND_TOOL_TOKENS_FIELD] {
        assert!(
            doc.contains(name),
            "`{name}` is recorded on every round span and undocumented, so \
             nobody reading docs/logging.md knows it exists"
        );
    }
    for metric in [
        PROMPT_PART_TOKENS_METRIC,
        PROMPT_TOOLS_METRIC,
        PROMPT_MEASURED_METRIC,
        PROMPT_TOOL_SERVER_TOKENS_METRIC,
        PROMPT_ROUND_TOOLS_METRIC,
        PROMPT_ROUND_MEASURED_METRIC,
    ] {
        assert!(
            doc.contains(metric),
            "`{metric}` is missing from the metrics table in docs/logging.md"
        );
    }
    for claim in [
        "will not sum exactly to the provider's own input count",
        "estimate",
    ] {
        assert!(
            doc.contains(claim),
            "docs/logging.md must say {claim:?}: a figure presented without \
             its accuracy claim reads as a measurement"
        );
    }
}

// ---------------------------------------------------------------------------
// #1212: the tool block, measured per round and per server.
//
// #1203 records the breakdown once per turn, on the first assembly. Activation
// only ever adds within a turn, so round one reports the floor of exactly the
// growth #1212 bounds - the instrument could show neither the defect nor the
// fix. These are the two axes it was missing.
// ---------------------------------------------------------------------------

/// What one round says its own tool block cost.
const ROUND_TOOL_COUNT_FIELD: &str = "prompt.tool_count";
const ROUND_TOOL_TOKENS_FIELD: &str = "prompt.tool_schema_tokens";

/// The largest tool block any round of the turn sent, on the turn span.
const PROMPT_TOOL_COUNT_PEAK_FIELD: &str = "prompt.tool_count_max";
const PROMPT_TOOL_TOKENS_PEAK_FIELD: &str = "prompt.tool_schema_tokens_max";

const PROMPT_TOOL_SERVER_TOKENS_METRIC: &str = "llm.prompt.tool.tokens";
const PROMPT_ROUND_TOOLS_METRIC: &str = "llm.prompt.round.tools";
const PROMPT_ROUND_MEASURED_METRIC: &str = "llm.prompt.round.measured";

#[test]
fn each_round_span_carries_the_tool_count_and_schema_cost_of_its_own_block() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    let rounds = captured.spans_named("turn.round");
    assert_eq!(rounds.len(), 2, "the script runs two rounds");
    for round in rounds {
        assert_eq!(
            round.field(ROUND_TOOL_COUNT_FIELD),
            Some("1"),
            "every round advertised the one scripted tool, and every round has \
             to say so - the turn-level figure is only the first round's, which \
             is the floor of a set that grows; the round recorded {:?}",
            round.fields
        );
        let tokens = round
            .field(ROUND_TOOL_TOKENS_FIELD)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or_else(|| {
                panic!(
                    "the round carries no `{ROUND_TOOL_TOKENS_FIELD}`; it \
                     recorded {:?}",
                    round.fields
                )
            });
        assert!(tokens > 0, "an advertised schema costs prompt tokens");
    }
}

#[test]
fn the_turn_span_reports_the_grown_tool_block_and_not_only_its_opening() {
    let _serialised = serialised();
    // Round one advertises the discovery tool alone; its search activates a
    // fleet tool, so round two advertises two. The turn's opening figure is the
    // floor of that growth, which is why it cannot be the figure an operator
    // reads for how large the block got.
    let captured = run(Level::INFO, search_script(), ScriptedTools::searching());
    let turn = captured.span("turn");

    assert_eq!(
        prompt_field(turn, PROMPT_TOOL_COUNT_FIELD),
        1,
        "precondition: the turn opened with one advertised tool; it recorded {:?}",
        turn.fields
    );
    assert_eq!(
        prompt_field(turn, PROMPT_TOOL_COUNT_PEAK_FIELD),
        2,
        "the search activated a second tool, and the peak is what says so; the \
         turn recorded {:?}",
        turn.fields
    );
    assert!(
        prompt_field(turn, PROMPT_TOOL_TOKENS_PEAK_FIELD)
            > prompt_field(turn, "prompt.tool_schema_tokens"),
        "the larger block cost more, and the peak carries that round's own \
         cost; the turn recorded {:?}",
        turn.fields
    );
}

#[test]
fn the_tool_schema_cost_is_counted_per_server_so_an_operator_can_see_which_to_drop() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    // The scripted executor's tool is a daemon built-in, so that is the axis
    // value the turn must attribute its schema bill to. A single aggregate
    // says 23.7k and names nothing to drop.
    let by_server = captured.counter_delta(PROMPT_TOOL_SERVER_TOKENS_METRIC, &["daemon"]);
    assert!(
        by_server > 0,
        "`{PROMPT_TOOL_SERVER_TOKENS_METRIC}` must carry a per-server axis; \
         the run recorded {:?}",
        captured
            .after
            .counters
            .iter()
            .map(|c| (c.name, render_labels(&c.labels)))
            .collect::<Vec<_>>()
    );
    assert!(
        captured.counter_delta(PROMPT_ROUND_MEASURED_METRIC, &[]) >= 2,
        "the per-round denominator counts every round, not every turn"
    );
    assert_eq!(
        captured.counter_delta(PROMPT_ROUND_TOOLS_METRIC, &[]),
        2,
        "two rounds advertising one tool each is two, not one"
    );
}
