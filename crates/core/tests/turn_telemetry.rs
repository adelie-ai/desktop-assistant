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
    Conversation, ConversationId, ConversationSummary, Message, ToolCall, ToolDefinition,
    ToolNamespace,
};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ReasoningConfig, TokenUsage,
};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::ports::tools::ToolExecutor;
use desktop_assistant_core::ports::turn_telemetry::{TurnRoute, with_request_id, with_turn_route};
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

/// The conversation id used by turns that check label bounding. Distinctive
/// enough that finding it in a label value is unambiguous.
const CONVERSATION_ID: &str = "conv-SENTINEL-UNBOUNDED-CONVERSATION-ID";

/// The user id used by the same turns, for the same reason.
const USER_ID: &str = "user-SENTINEL-UNBOUNDED-USER-ID";

/// The correlation id the transport would have minted.
const REQUEST_ID: &str = "11111111-2222-4333-8444-555555555555";

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
        .map(|h| ((h.name.to_string(), render_labels(&h.labels)), h.total.count))
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
fn counter_delta(
    before: &Summary,
    after: &Summary,
    name: &str,
    label_contains: &[&str],
) -> u64 {
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
}

impl SpanRecord {
    fn field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }
}

/// A `tracing` layer that keeps every span it sees, with its fields and its
/// parent, so a test can read back what a span recorded.
#[derive(Clone, Default)]
struct SpanCapture(Arc<Mutex<Vec<SpanRecord>>>);

impl SpanCapture {
    fn spans(&self) -> Vec<SpanRecord> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
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
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(SpanRecord {
                id: id.clone(),
                name: attrs.metadata().name(),
                parent,
                fields,
            });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut spans = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // Newest first: span ids are reused once a span closes, so an older
        // closed span can share this id.
        if let Some(span) = spans.iter_mut().rev().find(|s| s.id == *id) {
            values.record(&mut FieldVisitor(&mut span.fields));
        }
    }
}

/// Everything one captured run produced.
struct Captured {
    console: String,
    spans: Vec<SpanRecord>,
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

    Captured {
        console: console.text(),
        spans: spans.spans(),
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

/// An LLM that replays a script, so the turn's shape is fixed by the test.
struct ScriptedLlm {
    responses: Mutex<Vec<LlmResponse>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
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
        let response = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Ok(LlmResponse::text("done"));
            }
            responses.remove(0)
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
}

impl ScriptedTools {
    fn ok() -> Self {
        Self {
            tools: vec![write_note()],
            failure: None,
        }
    }

    fn failing() -> Self {
        Self {
            tools: vec![write_note()],
            failure: Some("the tool refused".to_string()),
        }
    }
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
        Ok(self.tools.iter().find(|t| t.name == name).cloned())
    }

    async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        Vec::new()
    }

    async fn execute_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        match &self.failure {
            Some(message) => Err(CoreError::ToolExecution(message.clone())),
            None => Ok("ok".to_string()),
        }
    }
}

fn handler(
    responses: Vec<LlmResponse>,
    tools: ScriptedTools,
) -> ConversationHandler<MemStore, ScriptedLlm, ScriptedTools> {
    ConversationHandler::with_tools(
        MemStore::default(),
        ScriptedLlm::new(responses),
        tools,
        Box::new(|| CONVERSATION_ID.to_string()),
    )
}

fn usage(input: u64, output: u64) -> TokenUsage {
    TokenUsage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
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
fn three_round_script() -> Vec<LlmResponse> {
    vec![
        LlmResponse::with_tool_calls("", vec![tool_call("c1")]).with_usage(usage(100, 10)),
        LlmResponse::with_tool_calls("", vec![tool_call("c2")]).with_usage(usage(200, 20)),
        LlmResponse::text(REPLY_SENTINEL).with_usage(usage(300, 30)),
    ]
}

/// One tool round then an answer.
fn two_round_script() -> Vec<LlmResponse> {
    vec![
        LlmResponse::with_tool_calls("", vec![tool_call("c1")]).with_usage(usage(100, 10)),
        LlmResponse::text(REPLY_SENTINEL).with_usage(usage(200, 20)),
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
                handler
                    .send_prompt_with_override(
                        &conv.id,
                        PROMPT_SENTINEL.to_string(),
                        None,
                        String::new(),
                        Box::new(|_| true),
                        Box::new(|_| {}),
                        CancellationToken::new(),
                    )
                    .await
                    .expect("the turn completes");
            }),
        ),
    )
    .await;
}

fn run(level: Level, responses: Vec<LlmResponse>, tools: ScriptedTools) -> Captured {
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
            line.contains(field.as_ref() as &str),
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
        LlmResponse::text(REPLY_SENTINEL).with_usage(TokenUsage {
            input_tokens: Some(100),
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        }),
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
        LlmResponse::text(REPLY_SENTINEL).with_usage(TokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(10),
            cache_creation_input_tokens: Some(40),
            cache_read_input_tokens: Some(4_000),
        }),
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
            &[&format!("model={MODEL}"), &format!("provider={PROVIDER}")]
        ),
        2,
        "each round's provider call is measured by provider and model, so a \
         slow provider is attributable without reproducing the turn"
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
// The level contract (epic D10), on the new surface.
// ---------------------------------------------------------------------------

#[test]
fn turn_span_records_no_content() {
    let _serialised = serialised();
    let captured = run(Level::INFO, two_round_script(), ScriptedTools::ok());

    // Span fields first: an `#[instrument]` without `skip` captures its
    // arguments, and nothing prints them, so this is invisible on the console
    // and still exports over OTLP.
    for span in &captured.spans {
        for (key, value) in &span.fields {
            for sentinel in [PROMPT_SENTINEL, TOOL_ARGUMENT_SENTINEL, REPLY_SENTINEL] {
                assert!(
                    !value.contains(sentinel),
                    "span `{}` field `{key}` carries content: {value}",
                    span.name
                );
            }
        }
    }

    // Then events, which the console does show.
    for sentinel in [PROMPT_SENTINEL, TOOL_ARGUMENT_SENTINEL, REPLY_SENTINEL] {
        assert!(
            !captured.console.contains(sentinel),
            "`{sentinel}` reached an INFO line\n--- console ---\n{}",
            captured.console
        );
    }
}

#[test]
fn the_content_test_can_see_content_when_there_is_some() {
    let _serialised = serialised();
    // The positive control for `turn_span_records_no_content`. Without it that
    // test cannot tell "nothing leaked" from "nothing ran": tool arguments are
    // logged at DEBUG deliberately, so they must be visible when asked for.
    let captured = run(Level::TRACE, two_round_script(), ScriptedTools::ok());

    assert!(
        captured.console.contains(TOOL_ARGUMENT_SENTINEL),
        "tool arguments belong at DEBUG, so an operator who needs them can ask\n\
         --- console ---\n{}",
        captured.console
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
    assert!(
        !module.exists(),
        "{} still exists",
        module.display()
    );
}
