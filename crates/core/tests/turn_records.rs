//! What a turn wrote down about itself (issue #1252).
//!
//! A person asks why the assistant did something. The answer is in the bytes
//! the model was shown, and those bytes exist in memory for one provider call
//! and are then gone. These tests are that promise reduced to this crate: a
//! turn hands its recorder the request exactly as the connector got it, the
//! reply exactly as it came back, and every tool call with the result it
//! produced.
//!
//! ## Why the connector is the witness
//!
//! [`CapturingLlm`] keeps every `Vec<Message>` it is handed. The assertions
//! then compare the stored record against what the connector actually
//! received, rather than against a prompt this file assembled - a test that
//! builds its own expectation proves the test's idea of the prompt, not the
//! daemon's.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, Message, Role, ToolCall, ToolDefinition,
};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ReasoningConfig, TokenUsage,
};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::ports::tools::ToolExecutor;
use desktop_assistant_core::ports::turn_record::{
    RoundRecord, RoundToolResults, TurnRecord, TurnRecorder,
};
use desktop_assistant_core::ports::turn_telemetry::{
    TurnRoute, resolve_turn_trace, with_request_id, with_turn_route, with_turn_trace,
};
use desktop_assistant_core::service::ConversationHandler;
use tokio_util::sync::CancellationToken;

/// The user's own words, which must appear in the recorded request verbatim.
const PROMPT: &str = "TURN-RECORD-PROMPT-what-did-you-see";

/// The model's own reply text.
const REPLY: &str = "TURN-RECORD-REPLY-here-is-what-I-concluded";

/// A tool call's arguments, chosen by the model.
const TOOL_ARGUMENTS: &str = r#"{"note":"TURN-RECORD-ARGUMENT"}"#;

/// What the tool returned.
const TOOL_OUTPUT: &str = "TURN-RECORD-RESULT-the-file-this-tool-read";

const TOOL_NAME: &str = "write_note";
const CALL_ID: &str = "call-1";
const USER_ID: &str = "turn-record-user";
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
// The recorder under test.
// ---------------------------------------------------------------------------

/// Everything a turn handed its recorder, in the order it was handed over.
#[derive(Default)]
struct Recorded {
    turns: Vec<TurnRecord>,
    rounds: Vec<RoundRecord>,
    results: Vec<RoundToolResults>,
}

#[derive(Default, Clone)]
struct FakeRecorder(Arc<Mutex<Recorded>>);

impl FakeRecorder {
    fn seen(&self) -> std::sync::MutexGuard<'_, Recorded> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[async_trait::async_trait]
impl TurnRecorder for FakeRecorder {
    async fn record_turn(&self, turn: TurnRecord) -> Result<(), CoreError> {
        self.seen().turns.push(turn);
        Ok(())
    }

    async fn record_round(&self, round: RoundRecord) -> Result<(), CoreError> {
        self.seen().rounds.push(round);
        Ok(())
    }

    async fn record_round_results(&self, results: RoundToolResults) -> Result<(), CoreError> {
        self.seen().results.push(results);
        Ok(())
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

/// An LLM that replays a script and keeps every request it was handed.
#[derive(Default)]
struct CapturingLlm {
    responses: Mutex<Vec<LlmResponse>>,
    /// One entry per call, in call order: the messages the connector received.
    requests: Mutex<Vec<Vec<Message>>>,
}

impl CapturingLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<Vec<Message>> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl LlmClient for CapturingLlm {
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        _tools: &[ToolDefinition],
        _reasoning: ReasoningConfig,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        self.requests.lock().unwrap().push(messages);
        let response = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                LlmResponse::text(REPLY)
            } else {
                responses.remove(0)
            }
        };
        if !response.text.is_empty() {
            on_chunk(response.text.clone());
        }
        Ok(response)
    }
}

/// What a provider says when it cannot serve the call.
const PROVIDER_FAULT: &str = "TURN-RECORD-PROVIDER-IS-DOWN";

/// A connector that always fails, so a round's error path can be recorded.
struct FailingLlm;

#[async_trait::async_trait]
impl LlmClient for FailingLlm {
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: &[ToolDefinition],
        _reasoning: ReasoningConfig,
        _on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        Err(CoreError::Llm(PROVIDER_FAULT.to_string()))
    }
}

/// A connector that never stops asking for tool calls, so the turn spends its
/// whole round budget and ends in the wind-down.
struct AlwaysCalling;

#[async_trait::async_trait]
impl LlmClient for AlwaysCalling {
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        _tools: &[ToolDefinition],
        _reasoning: ReasoningConfig,
        _on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // The wind-down offers no tools and carries its own instruction, so it
        // is the one call this connector answers with text.
        if messages
            .iter()
            .any(|m| m.content.contains(WIND_DOWN_MARKER))
        {
            return Ok(LlmResponse::text(REPLY));
        }
        Ok(LlmResponse::with_tool_calls(
            "",
            vec![ToolCall::new(CALL_ID, TOOL_NAME, TOOL_ARGUMENTS)],
        ))
    }
}

/// A phrase from the transient wrap-up instruction, which reaches the
/// connector on the wind-down call and on no other. Spelled out here rather
/// than imported, so a rewording of the instruction fails this test instead of
/// travelling with it.
const WIND_DOWN_MARKER: &str = "reached this turn's limit on tool";

/// A tool executor that trips the turn's cancellation token as it answers its
/// first call, so the round is abandoned from INSIDE its tool loop with one
/// call already resolved. A round whose cancellation lands between rounds
/// takes the ordinary end-of-round path instead and proves nothing about this.
struct CancellingTool(CancellationToken);

impl ToolExecutor for CancellingTool {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        vec![OneTool::definition()]
    }

    async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        Ok(vec![])
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        Ok((name == TOOL_NAME).then(OneTool::definition))
    }

    async fn execute_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        self.0.cancel();
        Ok(TOOL_OUTPUT.to_string())
    }
}

/// The inline cap this test runs the turn under, small enough that a modest
/// tool output is over it.
const HEAD_CAP: usize = 512;

/// The end of the oversized output, which a head can never reach.
const BIG_TOOL_TAIL: &str = "TURN-RECORD-TAIL-THE-MODEL-NEVER-SAW";

/// A tool whose output is well past the round's inline cap.
struct BigTool;

impl BigTool {
    fn output() -> String {
        format!("{}{BIG_TOOL_TAIL}", "x".repeat(HEAD_CAP * 4))
    }
}

impl ToolExecutor for BigTool {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        vec![OneTool::definition()]
    }

    async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        Ok(vec![])
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        Ok((name == TOOL_NAME).then(OneTool::definition))
    }

    async fn execute_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        Ok(Self::output())
    }
}

/// A tool executor that advertises one tool and answers it with fixed output.
struct OneTool;

impl OneTool {
    fn definition() -> ToolDefinition {
        ToolDefinition {
            name: TOOL_NAME.to_string(),
            description: "Write a note.".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }
}

impl ToolExecutor for OneTool {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        vec![Self::definition()]
    }

    async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        Ok(vec![])
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        Ok((name == TOOL_NAME).then(Self::definition))
    }

    async fn execute_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        Ok(TOOL_OUTPUT.to_string())
    }
}

// ---------------------------------------------------------------------------
// Driving one turn.
// ---------------------------------------------------------------------------

fn usage(input: u64, output: u64) -> TokenUsage {
    TokenUsage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    }
}

/// A script that calls one tool, then answers.
fn tool_then_answer() -> Vec<LlmResponse> {
    vec![
        LlmResponse::with_tool_calls("", vec![ToolCall::new(CALL_ID, TOOL_NAME, TOOL_ARGUMENTS)])
            .with_usage(usage(100, 10)),
        LlmResponse::text(REPLY).with_usage(usage(200, 20)),
    ]
}

/// What one turn produced: the recorder's contents and the connector's own
/// view of what it was sent.
struct Ran {
    recorded: FakeRecorder,
    llm: Arc<CapturingLlm>,
    conversation_id: String,
}

/// Run one turn against `responses`, with the turn recorder wired.
async fn run_turn(responses: Vec<LlmResponse>) -> Ran {
    let recorded = FakeRecorder::default();
    let llm = Arc::new(CapturingLlm::new(responses));
    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::clone(&llm),
        OneTool,
        Box::new(|| "conv-turn-record".to_string()),
    )
    .with_turn_recorder(Arc::new(recorded.clone()));

    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    let conversation_id = conv.id.0.clone();
    let trace = resolve_turn_trace(None, REQUEST_ID, &conversation_id);
    with_turn_trace(
        Some(trace),
        with_user_id(
            UserId::new(USER_ID),
            with_request_id(
                REQUEST_ID.to_string(),
                with_turn_route(route(), async {
                    handler
                        .send_prompt(
                            &conv.id,
                            PROMPT.to_string(),
                            Box::new(|_| true),
                            Box::new(|_| {}),
                        )
                        .await
                        .expect("the turn answers");
                }),
            ),
        ),
    )
    .await;

    Ran {
        recorded,
        llm,
        conversation_id,
    }
}

// ---------------------------------------------------------------------------
// Acceptance.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_turn_record_holds_every_round_of_the_turn() {
    let ran = run_turn(tool_then_answer()).await;
    let seen = ran.recorded.seen();

    assert_eq!(
        seen.turns.len(),
        1,
        "one turn writes one turn record; got {:?}",
        seen.turns
    );
    let turn = &seen.turns[0];
    assert_eq!(turn.correlation_id, REQUEST_ID);
    assert_eq!(turn.conversation_id, ran.conversation_id);
    assert_eq!(turn.connection_id.as_deref(), Some(CONNECTION_ID));
    assert_eq!(turn.provider.as_deref(), Some(PROVIDER));
    assert_eq!(turn.model.as_deref(), Some(MODEL));
    assert!(
        !turn.tool_policy.is_empty(),
        "the turn record states the tool policy it resolved to"
    );

    assert_eq!(
        seen.rounds.len(),
        2,
        "a turn that called a tool and then answered ran two rounds; got {:?}",
        seen.rounds.iter().map(|r| r.round).collect::<Vec<_>>()
    );
    assert_eq!(
        seen.rounds.iter().map(|r| r.round).collect::<Vec<_>>(),
        vec![1, 2],
        "rounds are numbered one-based and in order, the same as the round span"
    );
    for round in &seen.rounds {
        assert_eq!(
            round.correlation_id, turn.correlation_id,
            "every round is filed under its turn"
        );
        assert_eq!(
            round.conversation_id, ran.conversation_id,
            "and carries its own conversation, so no record is unattributable"
        );
    }
    assert_eq!(
        seen.rounds[1].response_text, REPLY,
        "the last round holds the reply the person read"
    );
    assert_eq!(
        seen.rounds[0].usage.as_ref().and_then(|u| u.input_tokens),
        Some(100),
        "each round carries the token usage the provider reported for it"
    );
    assert_eq!(
        seen.rounds[1].usage.as_ref().and_then(|u| u.input_tokens),
        Some(200)
    );
}

#[tokio::test]
async fn a_round_record_reproduces_the_request_as_sent() {
    let ran = run_turn(tool_then_answer()).await;
    let sent = ran.llm.requests();
    let seen = ran.recorded.seen();

    // The turn's rounds are the connector's first calls. A conversation with no
    // title asks the model for one after the turn has answered, and that call
    // is not a round of anything - so the comparison is over the rounds the
    // record claims, not over every call the connector served.
    assert_eq!(
        seen.rounds.len(),
        2,
        "a turn that called a tool and then answered ran two rounds"
    );
    assert!(
        sent.len() >= seen.rounds.len(),
        "the connector served at least the rounds the record claims"
    );
    for (index, (sent, recorded)) in sent.iter().zip(seen.rounds.iter()).enumerate() {
        assert_eq!(
            recorded.request.len(),
            sent.len(),
            "round {} recorded {} messages and sent {}",
            index + 1,
            recorded.request.len(),
            sent.len()
        );
        for (position, (sent, recorded)) in sent.iter().zip(recorded.request.iter()).enumerate() {
            assert_eq!(
                (&recorded.role, recorded.content.as_str()),
                (&sent.role, sent.content.as_str()),
                "round {}, message {position}: the record must be the bytes the \
                 connector was handed, in the order it got them",
                index + 1
            );
        }
    }

    let first = &seen.rounds[0].request;
    assert_eq!(
        first.first().map(|m| &m.role),
        Some(&Role::System),
        "the system prompt is the first message and it is part of the record"
    );
    assert!(
        first
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("Adele")),
        "the recorded system prompt is the assembled one, not a placeholder"
    );
    assert!(
        first.iter().any(|m| m.content.contains(PROMPT)),
        "and the person's own words are in it"
    );
}

#[tokio::test]
async fn tool_calls_and_results_are_recorded_with_their_round() {
    let ran = run_turn(tool_then_answer()).await;
    let seen = ran.recorded.seen();

    let calls = &seen.rounds[0].response_tool_calls;
    assert_eq!(calls.len(), 1, "the first round asked for one tool call");
    assert_eq!(calls[0].id, CALL_ID);
    assert_eq!(calls[0].name, TOOL_NAME);
    assert_eq!(
        calls[0].arguments, TOOL_ARGUMENTS,
        "the arguments are recorded as the model wrote them"
    );
    assert!(
        seen.rounds[1].response_tool_calls.is_empty(),
        "the answering round asked for none"
    );

    let results: Vec<&RoundToolResults> = seen.results.iter().collect();
    assert_eq!(
        results.len(),
        1,
        "one round dispatched tools, so one round's results were recorded; \
         got rounds {:?}",
        results.iter().map(|r| r.round).collect::<Vec<_>>()
    );
    let recorded = results[0];
    assert_eq!(
        recorded.round, 1,
        "the results are filed under the round whose calls produced them"
    );
    assert_eq!(recorded.correlation_id, REQUEST_ID);
    assert_eq!(recorded.conversation_id, ran.conversation_id);
    assert_eq!(recorded.results.len(), 1);
    let result = &recorded.results[0];
    assert_eq!(result.role, Role::Tool);
    assert_eq!(result.tool_call_id.as_deref(), Some(CALL_ID));
    assert_eq!(
        result.content, TOOL_OUTPUT,
        "the result is recorded as the turn stored it"
    );
}

#[tokio::test]
async fn a_round_records_its_own_results_when_a_model_reuses_a_call_id() {
    // A call id comes off the model's reply, so nothing stops one model
    // spelling every call `call-1`. Filing a round's results by id alone would
    // then make each round's record swallow every earlier round's, and the
    // record would read as a turn whose last round ran everything.
    let ran = run_turn(vec![
        LlmResponse::with_tool_calls("", vec![ToolCall::new(CALL_ID, TOOL_NAME, TOOL_ARGUMENTS)])
            .with_usage(usage(100, 10)),
        LlmResponse::with_tool_calls("", vec![ToolCall::new(CALL_ID, TOOL_NAME, TOOL_ARGUMENTS)])
            .with_usage(usage(150, 15)),
        LlmResponse::text(REPLY).with_usage(usage(200, 20)),
    ])
    .await;
    let seen = ran.recorded.seen();

    assert_eq!(
        seen.results.iter().map(|r| r.round).collect::<Vec<_>>(),
        vec![1, 2],
        "both tool rounds recorded their results"
    );
    for recorded in &seen.results {
        assert_eq!(
            recorded.results.len(),
            1,
            "round {} dispatched one call, so its record holds one result",
            recorded.round
        );
    }
}

#[tokio::test]
async fn a_round_that_failed_records_its_error_and_no_usage() {
    // The turn's most interesting rounds are the ones that did not work. A
    // record that only covers the happy path answers the question nobody
    // asks.
    let recorded = FakeRecorder::default();
    let llm = Arc::new(FailingLlm);
    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::clone(&llm),
        OneTool,
        Box::new(|| "conv-failing-round".to_string()),
    )
    .with_turn_recorder(Arc::new(recorded.clone()));
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    let _ = with_user_id(UserId::new(USER_ID), async {
        handler
            .send_prompt(
                &conv.id,
                PROMPT.to_string(),
                Box::new(|_| true),
                Box::new(|_| {}),
            )
            .await
    })
    .await;

    let seen = recorded.seen();
    assert_eq!(seen.rounds.len(), 1, "the failing round is still a round");
    let round = &seen.rounds[0];
    assert_eq!(
        round.error.as_deref().map(|e| e.contains(PROVIDER_FAULT)),
        Some(true),
        "the round names why it failed; got {:?}",
        round.error
    );
    assert!(
        round.usage.is_none(),
        "a call that did not answer reported no usage, so none is recorded"
    );
    assert!(
        round.response_text.is_empty() && round.response_tool_calls.is_empty(),
        "and nothing is invented in place of the reply it never gave"
    );
    assert!(
        !round.request.is_empty(),
        "the request it failed on is the whole point of recording it"
    );
}

#[tokio::test]
async fn an_oversized_tool_result_is_recorded_whole_while_the_round_read_its_head() {
    // The profiler this replaces stored a 200-character preview, and a preview
    // answers no question anybody asks of it. The two halves are separate
    // facts and both are recorded: `tool_results` holds every byte the tool
    // produced, and the next round's `request` holds the head the model was
    // actually shown. Recording either one twice would lose the other.
    let recorded = FakeRecorder::default();
    let llm = Arc::new(CapturingLlm::new(tool_then_answer()));
    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::clone(&llm),
        BigTool,
        Box::new(|| "conv-oversized".to_string()),
    )
    .with_turn_recorder(Arc::new(recorded.clone()))
    .with_max_tool_result_bytes(HEAD_CAP);
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    with_user_id(UserId::new(USER_ID), async {
        handler
            .send_prompt(
                &conv.id,
                PROMPT.to_string(),
                Box::new(|_| true),
                Box::new(|_| {}),
            )
            .await
            .expect("the turn answers");
    })
    .await;

    let seen = recorded.seen();
    let stored = &seen.results[0].results[0].content;
    assert_eq!(
        stored.len(),
        BigTool::output().len(),
        "the record holds every byte the tool produced, not a preview"
    );
    assert!(stored.ends_with(BIG_TOOL_TAIL), "including its tail");

    let second_round = &seen.rounds[1].request;
    let shown = second_round
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("the second round was shown the tool result");
    assert!(
        shown.content.len() < stored.len(),
        "and the request records the head the model was actually shown, which \
         is shorter: {} vs {}",
        shown.content.len(),
        stored.len()
    );
}

#[tokio::test]
async fn the_wind_down_that_closes_an_exhausted_turn_is_recorded() {
    // A turn that spends its whole tool budget makes one more provider call,
    // and its reply is the closing the person actually reads. The request
    // behind it exists nowhere else: the wrap-up instruction it carries is
    // dropped before the reply is persisted. So it is the one round whose
    // absence from the record could not be recovered from anywhere.
    let recorded = FakeRecorder::default();
    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::new(AlwaysCalling),
        OneTool,
        Box::new(|| "conv-wind-down".to_string()),
    )
    .with_turn_recorder(Arc::new(recorded.clone()));
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    let closing = with_user_id(UserId::new(USER_ID), async {
        handler
            .send_prompt(
                &conv.id,
                PROMPT.to_string(),
                Box::new(|_| true),
                Box::new(|_| {}),
            )
            .await
    })
    .await
    .expect("an exhausted turn still answers");
    assert_eq!(closing, REPLY, "the person read the wind-down's reply");

    let seen = recorded.seen();
    let last = seen
        .rounds
        .last()
        .expect("the turn recorded at least one round");
    assert!(
        last.round > seen.rounds.len() as u32 - 1,
        "the wind-down is filed past the loop's last round; got {}",
        last.round
    );
    assert_eq!(
        last.response_text, REPLY,
        "and it holds the closing the person read"
    );
    assert!(
        last.request
            .iter()
            .any(|m| m.content.contains(WIND_DOWN_MARKER)),
        "with the request that produced it, wrap-up instruction included - the \
         one prompt that exists nowhere else"
    );
}

#[tokio::test]
async fn a_cancelled_round_records_the_calls_that_already_ran() {
    // The calls that ran committed their side effects. A record saying no tool
    // ran is a wrong answer, which is worse than a missing one.
    //
    // Two calls in the round, deliberately: the first resolves and trips the
    // token, so the per-tool checkpoint fires before the second and the round
    // leaves from inside its own tool loop. One call would let the round reach
    // its ordinary end and record there, which is a different path.
    let recorded = FakeRecorder::default();
    let token = CancellationToken::new();
    let two_calls = vec![
        LlmResponse::with_tool_calls(
            "",
            vec![
                ToolCall::new(CALL_ID, TOOL_NAME, TOOL_ARGUMENTS),
                ToolCall::new("call-2", TOOL_NAME, TOOL_ARGUMENTS),
            ],
        )
        .with_usage(usage(100, 10)),
    ];
    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::new(CapturingLlm::new(two_calls)),
        CancellingTool(token.clone()),
        Box::new(|| "conv-cancelled".to_string()),
    )
    .with_turn_recorder(Arc::new(recorded.clone()));
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    let outcome = with_user_id(UserId::new(USER_ID), async {
        handler
            .send_prompt_with_override(
                &conv.id,
                PROMPT.to_string(),
                None,
                String::new(),
                Box::new(|_| true),
                Box::new(|_| {}),
                token,
            )
            .await
    })
    .await;
    assert!(
        matches!(outcome, Err(CoreError::Cancelled)),
        "the turn was cancelled"
    );

    let seen = recorded.seen();
    let results = seen
        .results
        .first()
        .expect("the cancelled round recorded what it had run");
    assert_eq!(results.round, 1);
    assert_eq!(
        results.results.len(),
        1,
        "the first call resolved and the second never ran, so the record holds \
         one result rather than none or two"
    );
    assert_eq!(results.results[0].content, TOOL_OUTPUT);
    assert_eq!(results.results[0].tool_call_id.as_deref(), Some(CALL_ID));
}

#[tokio::test]
async fn capture_changes_nothing_the_model_sees() {
    // The other half of the switch, and the property that lets it default on:
    // a captured turn must send the provider exactly what an uncaptured one
    // sends. A record that alters the thing it records is worse than no
    // record, because it is read as evidence.
    let captured = run_turn(tool_then_answer()).await;

    let llm = Arc::new(CapturingLlm::new(tool_then_answer()));
    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::clone(&llm),
        OneTool,
        Box::new(|| "conv-turn-record".to_string()),
    );
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    let trace = resolve_turn_trace(None, REQUEST_ID, &conv.id.0);
    let answer = with_turn_trace(
        Some(trace),
        with_user_id(
            UserId::new(USER_ID),
            with_request_id(
                REQUEST_ID.to_string(),
                with_turn_route(route(), async {
                    handler
                        .send_prompt(
                            &conv.id,
                            PROMPT.to_string(),
                            Box::new(|_| true),
                            Box::new(|_| {}),
                        )
                        .await
                }),
            ),
        ),
    )
    .await
    .expect("the turn answers with no recorder wired");

    assert_eq!(answer, REPLY);
    assert_eq!(
        llm.requests(),
        captured.llm.requests(),
        "an unrecorded turn and a recorded one send the provider the same bytes"
    );
}

#[tokio::test]
async fn a_failing_recorder_does_not_fail_the_turn() {
    // A debugging record is worth less than the answer a person asked for.
    struct Failing;

    #[async_trait::async_trait]
    impl TurnRecorder for Failing {
        async fn record_turn(&self, _turn: TurnRecord) -> Result<(), CoreError> {
            Err(CoreError::Storage("no database".to_string()))
        }

        async fn record_round(&self, _round: RoundRecord) -> Result<(), CoreError> {
            Err(CoreError::Storage("no database".to_string()))
        }

        async fn record_round_results(&self, _results: RoundToolResults) -> Result<(), CoreError> {
            Err(CoreError::Storage("no database".to_string()))
        }
    }

    let handler = ConversationHandler::with_tools(
        MemStore::default(),
        Arc::new(CapturingLlm::new(tool_then_answer())),
        OneTool,
        Box::new(|| "conv-failing-recorder".to_string()),
    )
    .with_turn_recorder(Arc::new(Failing));
    let conv = handler
        .create_conversation("c".into(), vec![])
        .await
        .expect("create the conversation");
    let answer = with_user_id(UserId::new(USER_ID), async {
        handler
            .send_prompt(
                &conv.id,
                PROMPT.to_string(),
                Box::new(|_| true),
                Box::new(|_| {}),
            )
            .await
    })
    .await
    .expect("a store that cannot be written must not end the turn");
    assert_eq!(answer, REPLY);
}
