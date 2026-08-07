//! The level contract for the turn loop.
//!
//! > INFO carries ids, counts, durations, model names and token counts. Never
//! > content.
//! > DEBUG carries prompts, the full assembled context, and tool arguments.
//!
//! The turn loop used to write every tool call's arguments, and the head of a
//! model reply that sanitized to nothing, at a level every shipped deployment
//! turns on. Both are conversation content, and both reached the journal and
//! the cluster log stack, which have no per-user scoping and no deletion
//! story. These tests hold the line in both directions: the content is absent
//! at INFO, and still present at DEBUG so the operator who needs it can ask
//! for it.

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex, Once};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, Message, ToolCall, ToolDefinition,
    ToolNamespace,
};
use desktop_assistant_core::ports::client_tools::{ClientToolPort, with_client_tools};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse, ReasoningConfig};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::ports::tools::ToolExecutor;
use desktop_assistant_core::service::ConversationHandler;
use tokio_util::sync::CancellationToken;
use tracing::Level;

/// A string that only ever appears inside a tool call's arguments. Shaped like
/// the two things the finding named: a credential and a personal fact.
const TOOL_ARGUMENT_SENTINEL: &str = "sk-live-PEANUT-ALLERGY-SCHOOL-NAME";

/// A string that only ever appears inside model text that sanitizes away.
const MODEL_TEXT_SENTINEL: &str = "SENTINEL-MODEL-REASONING-ABOUT-THE-USER";

// ---------------------------------------------------------------------------
// Log capture.
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().expect("lock the capture buffer").clone())
            .expect("captured log output is UTF-8")
    }
}

impl io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("lock the capture buffer").extend(buf);
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
/// `tracing` caches each callsite's interest globally, not per thread. Without
/// a permissive global default, a callsite first evaluated on a thread running
/// the INFO-capped test can latch "never" for the whole process, and the
/// DEBUG-capped test then never sees it. That is a scheduling-dependent flake,
/// not a real failure.
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

/// Drive `future` to completion with every record at `level` or above captured.
///
/// A current-thread runtime keeps the whole run on the thread that holds the
/// thread-local subscriber, so the capture covers the turn and not only its
/// first poll.
fn capture_at<F: std::future::Future>(level: Level, future: F) -> (F::Output, String) {
    ensure_permissive_global_default();
    let captured = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(captured.clone())
        .with_ansi(false)
        .finish();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a current-thread runtime");
    let output = tracing::subscriber::with_default(subscriber, || runtime.block_on(future));
    (output, captured.text())
}

// ---------------------------------------------------------------------------
// Stubs. The turn has to be real; nothing it talks to does.
// ---------------------------------------------------------------------------

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

struct ScriptedToolExecutor {
    tools: Vec<ToolDefinition>,
    /// When set, every dispatch fails with this error instead of succeeding.
    failure: Option<String>,
}

impl ToolExecutor for ScriptedToolExecutor {
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

/// A client-tool port whose every call fails with `failure`.
///
/// The turn loop routes a registered name here instead of to the server-side
/// executor, and that arm has its own log site, so it needs its own coverage.
struct FailingClientToolPort {
    tools: Vec<ToolDefinition>,
    failure: String,
}

#[async_trait::async_trait]
impl ClientToolPort for FailingClientToolPort {
    async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.clone()
    }

    async fn is_registered(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        _tool_name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        Err(CoreError::ToolExecution(self.failure.clone()))
    }
}

/// A handler whose every tool dispatch fails with `failure`.
fn failing_handler(
    failure: &str,
) -> ConversationHandler<MemStore, ScriptedLlm, ScriptedToolExecutor> {
    ConversationHandler::with_tools(
        MemStore {
            data: Mutex::new(HashMap::new()),
        },
        ScriptedLlm {
            responses: Mutex::new(tool_call_script()),
        },
        ScriptedToolExecutor {
            tools: vec![ToolDefinition::new(
                "write_note",
                "write a note",
                serde_json::json!({"type": "object"}),
            )],
            failure: Some(failure.to_string()),
        },
        Box::new(|| "conv-1".to_string()),
    )
}

fn handler(
    responses: Vec<LlmResponse>,
) -> ConversationHandler<MemStore, ScriptedLlm, ScriptedToolExecutor> {
    ConversationHandler::with_tools(
        MemStore {
            data: Mutex::new(HashMap::new()),
        },
        ScriptedLlm {
            responses: Mutex::new(responses),
        },
        ScriptedToolExecutor {
            tools: vec![ToolDefinition::new(
                "write_note",
                "write a note",
                serde_json::json!({"type": "object"}),
            )],
            failure: None,
        },
        Box::new(|| "conv-1".to_string()),
    )
}

/// Run one turn in which the model calls a tool whose arguments carry the
/// sentinel, then answers.
async fn turn_with_a_tool_call(
    handler: &ConversationHandler<MemStore, ScriptedLlm, ScriptedToolExecutor>,
) {
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    handler
        .send_prompt_with_override(
            &conv.id,
            "go".into(),
            None,
            String::new(),
            Box::new(|_| true),
            Box::new(|_| {}),
            CancellationToken::new(),
        )
        .await
        .expect("the turn completes");
}

fn tool_call_script() -> Vec<LlmResponse> {
    let arguments = serde_json::json!({ "note": TOOL_ARGUMENT_SENTINEL }).to_string();
    vec![
        LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "write_note", arguments)]),
        LlmResponse::text("saved"),
    ]
}

// ---------------------------------------------------------------------------
// The named criteria.
// ---------------------------------------------------------------------------

#[test]
fn no_content_at_info() {
    let (_, logs) = capture_at(Level::INFO, async {
        let handler = handler(tool_call_script());
        turn_with_a_tool_call(&handler).await;
    });

    assert!(
        !logs.contains(TOOL_ARGUMENT_SENTINEL),
        "a tool call's arguments are conversation content and must not reach an INFO line\n\
         --- captured at INFO ---\n{logs}"
    );
    assert!(
        logs.contains("executing tool"),
        "the INFO line itself must survive - the fix is to drop the content, not the line\n\
         --- captured at INFO ---\n{logs}"
    );
}

#[test]
fn content_appears_at_debug() {
    let (_, logs) = capture_at(Level::DEBUG, async {
        let handler = handler(tool_call_script());
        turn_with_a_tool_call(&handler).await;
    });

    assert!(
        logs.contains(TOOL_ARGUMENT_SENTINEL),
        "tool arguments belong at DEBUG, so an operator who needs them can ask\n\
         --- captured at DEBUG ---\n{logs}"
    );
}

#[test]
fn no_model_text_at_info_when_the_reply_sanitizes_to_nothing() {
    // A reply that is entirely a thinking block sanitizes to nothing, which is
    // what the "empty visible text" warning reports. The warning used to carry
    // the first hundred characters of the raw reply, which is the model's own
    // reasoning about the user.
    let raw = format!("<think>{MODEL_TEXT_SENTINEL}</think>");
    let (_, logs) = capture_at(Level::INFO, async move {
        let handler = handler(vec![LlmResponse::text(raw)]);
        turn_with_a_tool_call(&handler).await;
    });

    assert!(
        !logs.contains(MODEL_TEXT_SENTINEL),
        "the head of a model reply is content and must not reach a WARN line\n\
         --- captured at INFO ---\n{logs}"
    );
    assert!(
        logs.contains("empty visible text"),
        "the warning itself must survive - it is how an operator sees the condition\n\
         --- captured at INFO ---\n{logs}"
    );
}

// ---------------------------------------------------------------------------
// A failing tool's own words.
//
// The success arm puts a tool's output at DEBUG, so tool output is already
// treated as content. The failure arm is the same content by another route: an
// MCP server says what it could not do, and that sentence quotes the argument
// it was given. `McpError::ServerError` renders the server's message verbatim,
// so "failed to read <path>: permission denied" arrives at the log site whole.
// WARN is above INFO, so it is on in every deployment.
// ---------------------------------------------------------------------------

/// Shaped like what a file or shell tool actually says when it fails.
const TOOL_ERROR_SENTINEL: &str =
    "failed to read /home/example/.ssh/SENTINEL-ID-ED25519: permission denied";

/// Run one turn whose tool call fails inside the server-side executor.
fn server_tool_failure_capturing(level: Level) -> String {
    let (_, logs) = capture_at(level, async {
        let handler = failing_handler(TOOL_ERROR_SENTINEL);
        turn_with_a_tool_call(&handler).await;
    });
    logs
}

/// Run one turn whose tool call is routed to a client tool that fails.
fn client_tool_failure_capturing(level: Level) -> String {
    let (_, logs) = capture_at(level, async {
        let handler = handler(tool_call_script());
        let port: std::sync::Arc<dyn ClientToolPort> = std::sync::Arc::new(FailingClientToolPort {
            tools: vec![ToolDefinition::new(
                "write_note",
                "write a note",
                serde_json::json!({"type": "object"}),
            )],
            failure: TOOL_ERROR_SENTINEL.to_string(),
        });
        with_client_tools(port, async {
            turn_with_a_tool_call(&handler).await;
        })
        .await;
    });
    logs
}

#[test]
fn no_tool_error_text_at_info() {
    let logs = server_tool_failure_capturing(Level::INFO);
    assert!(
        !logs.contains(TOOL_ERROR_SENTINEL),
        "a failing tool's message quotes what it failed on and must not reach a WARN line\n\
         --- captured at INFO ---\n{logs}"
    );
    // The line itself must survive, and must still say which tool and what
    // went wrong - otherwise deleting the log site would pass this test.
    assert!(
        logs.contains("tool execution failed"),
        "the failure line must survive\n--- captured at INFO ---\n{logs}"
    );
    assert!(
        logs.contains(r#"error_kind="tool_execution""#),
        "the line must still classify the failure\n--- captured at INFO ---\n{logs}"
    );
}

#[test]
fn no_client_tool_error_text_at_info() {
    let logs = client_tool_failure_capturing(Level::INFO);
    assert!(
        !logs.contains(TOOL_ERROR_SENTINEL),
        "the client-tool arm has its own log site and the same rule applies\n\
         --- captured at INFO ---\n{logs}"
    );
    assert!(
        logs.contains("client tool execution failed"),
        "the failure line must survive\n--- captured at INFO ---\n{logs}"
    );
    assert!(
        logs.contains(r#"error_kind="tool_execution""#),
        "the line must still classify the failure\n--- captured at INFO ---\n{logs}"
    );
}

#[test]
fn tool_error_text_appears_at_debug() {
    for logs in [
        server_tool_failure_capturing(Level::DEBUG),
        client_tool_failure_capturing(Level::DEBUG),
    ] {
        assert!(
            logs.contains(TOOL_ERROR_SENTINEL),
            "the message belongs at DEBUG, so an operator diagnosing the tool can ask\n\
             --- captured at DEBUG ---\n{logs}"
        );
    }
}
