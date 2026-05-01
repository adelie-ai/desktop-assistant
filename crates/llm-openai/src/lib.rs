use std::collections::HashMap;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ReasoningConfig,
    TokenUsage, current_model_override,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio_stream::StreamExt;

/// Return the prompt-token context window for a known OpenAI model id.
///
/// OpenAI exposes context windows through `/v1/models` only inconsistently,
/// so this curated table covers the common families we ship with the model
/// picker. Returns `None` for ids that are unknown or that intentionally
/// defer to the daemon's universal fallback (e.g. the `gpt-5*` family,
/// whose window varies by sub-tier).
///
/// Order matters: more specific prefixes are checked before their general
/// counterparts (e.g. `gpt-4-turbo` before `gpt-4`).
pub fn context_limit_for_model(model: &str) -> Option<u64> {
    if model.starts_with("gpt-4o") || model.starts_with("gpt-4-turbo") {
        return Some(128_000);
    }
    if model.starts_with("gpt-4-32k") {
        return Some(32_768);
    }
    if model.starts_with("gpt-4") {
        return Some(8_192);
    }
    if model.starts_with("gpt-3.5-turbo-16k") {
        return Some(16_384);
    }
    if model.starts_with("gpt-3.5-turbo") {
        return Some(4_096);
    }
    if model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
        return Some(200_000);
    }
    None
}

/// OpenAI-compatible LLM client that streams completions via the Responses API.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    hosted_tool_search: bool,
}

impl OpenAiClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("gpt-5.4")
    }

    pub fn get_default_base_url() -> Option<&'static str> {
        Some("https://api.openai.com/v1")
    }

    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: Self::get_default_model().unwrap_or_default().to_string(),
            base_url: Self::get_default_base_url().unwrap_or_default().to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: false,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_temperature(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_hosted_tool_search(mut self, enabled: bool) -> Self {
        self.hosted_tool_search = enabled;
        self
    }

    /// Return the model name as the stable version identifier.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        Ok(self.model.clone())
    }

    /// Generate embeddings for a batch of texts.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("embedding HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "OpenAI embeddings API error (HTTP {status}): {body}"
            )));
        }

        let parsed: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse embedding response: {e}")))?;

        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    pub fn from_env() -> Result<Self, CoreError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| CoreError::Llm("OPENAI_API_KEY environment variable not set".into()))?;
        let mut client = Self::new(api_key);
        if let Ok(model) = std::env::var("OPENAI_MODEL") {
            client.model = model;
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            client.base_url = url;
        }
        Ok(client)
    }
}

// ---------------------------------------------------------------------------
// Responses API – request serialization types
// ---------------------------------------------------------------------------

/// OpenAI Responses-API reasoning block (`{ "effort": "low|medium|high" }`).
///
/// Emitted only for reasoning-capable models (see
/// [`model_supports_reasoning`]). For non-reasoning models the field is
/// omitted entirely; sending it there causes a 400 from the API.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct ReasoningBlock {
    effort: &'static str,
}

/// Unified request body for the Responses API (`POST /v1/responses`).
#[derive(Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningBlock>,
}

/// Heterogeneous input items for the Responses API.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
enum InputItem {
    Message(InputMessage),
    FunctionCall(InputFunctionCall),
    FunctionCallOutput(InputFunctionCallOutput),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct InputMessage {
    role: String,
    content: String,
}

/// A tool call from a previous assistant turn, replayed as input.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct InputFunctionCall {
    r#type: String,
    /// Synthetic ID for the input item (not the correlation key).
    id: String,
    /// The real correlation key matching `function_call_output`.
    call_id: String,
    name: String,
    arguments: String,
}

/// A tool result replayed as input.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct InputFunctionCallOutput {
    r#type: String,
    call_id: String,
    output: String,
}

/// Tool entry in the `tools` array – can be a function, namespace, or tool_search.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
enum ToolEntry {
    Function(FunctionTool),
    Namespace(NamespaceTool),
    ToolSearch(ToolSearchSentinel),
}

/// Flat function tool (name at top level, not nested under `function`).
#[derive(Serialize, Debug, Clone, PartialEq)]
struct FunctionTool {
    r#type: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    defer_loading: Option<bool>,
}

impl FunctionTool {
    fn from_definition(def: &ToolDefinition) -> Self {
        Self {
            r#type: "function".to_string(),
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
            defer_loading: None,
        }
    }

    fn from_definition_deferred(def: &ToolDefinition) -> Self {
        Self {
            r#type: "function".to_string(),
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
            defer_loading: Some(true),
        }
    }
}

/// A tool namespace entry for hosted tool search.
#[derive(Serialize, Debug, Clone, PartialEq)]
struct NamespaceTool {
    r#type: String,
    name: String,
    description: String,
    tools: Vec<FunctionTool>,
}

impl NamespaceTool {
    fn from_namespace(ns: &ToolNamespace) -> Self {
        // Prefix namespace names to avoid collisions with OpenAI reserved
        // namespaces (e.g. "terminal", "browser").
        let name = format!("adele_{}", ns.name);
        Self {
            r#type: "namespace".to_string(),
            name,
            description: ns.description.clone(),
            tools: ns
                .tools
                .iter()
                .map(FunctionTool::from_definition_deferred)
                .collect(),
        }
    }
}

/// The `tool_search` sentinel that enables OpenAI's hosted tool discovery.
#[derive(Serialize, Debug, Clone, PartialEq)]
struct ToolSearchSentinel {
    r#type: String,
}

// ---------------------------------------------------------------------------
// Embedding response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Responses API – streaming response types
// ---------------------------------------------------------------------------

/// `response.output_text.delta` event payload.
#[derive(Deserialize)]
struct TextDelta {
    delta: String,
}

/// `response.output_item.added` event payload.
#[derive(Deserialize)]
struct OutputItemAdded {
    output_index: usize,
    item: OutputItem,
}

/// An output item (we only care about `function_call` items).
#[derive(Deserialize)]
struct OutputItem {
    r#type: String,
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

/// `response.function_call_arguments.delta` event payload.
#[derive(Deserialize)]
struct FunctionArgsDelta {
    output_index: usize,
    delta: String,
}

/// `response.function_call_arguments.done` event payload.
#[derive(Deserialize)]
struct FunctionArgsDone {
    output_index: usize,
    arguments: String,
}

/// `response.completed` event payload.
#[derive(Deserialize)]
struct ResponseCompleted {
    response: ResponseCompletedInner,
}

#[derive(Deserialize)]
struct ResponseCompletedInner {
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

#[derive(Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

/// Entry in the tool call accumulator, keyed by `output_index`.
#[derive(Debug, Default)]
struct ResponseToolEntry {
    call_id: String,
    name: String,
    arguments: String,
}

/// Accumulator for building tool calls from Responses API streaming events.
#[derive(Debug, Default)]
struct ResponseToolAccumulator {
    entries: HashMap<usize, ResponseToolEntry>,
}

impl ResponseToolAccumulator {
    fn register(&mut self, output_index: usize, call_id: String, name: String) {
        self.entries.insert(
            output_index,
            ResponseToolEntry {
                call_id,
                name,
                arguments: String::new(),
            },
        );
    }

    fn append_arguments(&mut self, output_index: usize, delta: &str) {
        if let Some(entry) = self.entries.get_mut(&output_index) {
            entry.arguments.push_str(delta);
        }
    }

    fn finalize_arguments(&mut self, output_index: usize, arguments: &str) {
        if let Some(entry) = self.entries.get_mut(&output_index) {
            entry.arguments = arguments.to_string();
        }
    }

    fn into_tool_calls(self) -> Vec<ToolCall> {
        let mut pairs: Vec<(usize, ResponseToolEntry)> = self.entries.into_iter().collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        pairs
            .into_iter()
            .map(|(_, e)| ToolCall::new(e.call_id, e.name, e.arguments))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Message conversion: domain Messages → Responses API InputItems
// ---------------------------------------------------------------------------

/// Convert domain messages to Responses API input items plus an optional
/// `instructions` string (extracted from system messages; multiple system
/// messages are concatenated since the Responses API accepts a single string).
fn convert_messages(messages: &[Message]) -> (Vec<InputItem>, Option<String>) {
    let mut items = Vec::new();
    let mut instructions: Option<String> = None;

    for msg in messages {
        match msg.role {
            Role::System => match &mut instructions {
                Some(existing) => {
                    existing.push_str("\n\n");
                    existing.push_str(&msg.content);
                }
                None => {
                    instructions = Some(msg.content.clone());
                }
            },
            Role::User => {
                items.push(InputItem::Message(InputMessage {
                    role: "user".to_string(),
                    content: msg.content.clone(),
                }));
            }
            Role::Assistant => {
                // Emit text content as a message if non-empty
                if !msg.content.is_empty() {
                    items.push(InputItem::Message(InputMessage {
                        role: "assistant".to_string(),
                        content: msg.content.clone(),
                    }));
                }
                // Emit each tool call as a separate FunctionCall item
                for tc in &msg.tool_calls {
                    items.push(InputItem::FunctionCall(InputFunctionCall {
                        r#type: "function_call".to_string(),
                        id: format!("fc_{}", tc.id),
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    }));
                }
            }
            Role::Tool => {
                if let Some(call_id) = &msg.tool_call_id {
                    items.push(InputItem::FunctionCallOutput(InputFunctionCallOutput {
                        r#type: "function_call_output".to_string(),
                        call_id: call_id.clone(),
                        output: msg.content.clone(),
                    }));
                }
            }
        }
    }

    (items, instructions)
}

// ---------------------------------------------------------------------------
// Streaming implementation
// ---------------------------------------------------------------------------

impl OpenAiClient {
    /// Send a Responses API request and parse the SSE stream into an LlmResponse.
    async fn send_and_stream(
        &self,
        request_json: &str,
        request_body: &impl Serialize,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let request_bytes = request_json.len();
        tracing::info!(
            request_bytes,
            model = %self.model,
            "LLM request payload"
        );
        tracing::debug!(
            "LLM request body (first 2000 chars): {}",
            &request_json[..request_json.len().min(2000)]
        );

        let response = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request_body)
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "OpenAI API error (HTTP {status}): {body}"
            )));
        }

        let byte_stream = response.bytes_stream();
        let mapped_stream = byte_stream.map(|result: Result<bytes::Bytes, reqwest::Error>| {
            result.map_err(std::io::Error::other)
        });
        let stream_reader = tokio_util::io::StreamReader::new(mapped_stream);
        let mut lines = tokio::io::BufReader::new(stream_reader).lines();

        let mut full_response = String::new();
        let mut tool_acc = ResponseToolAccumulator::default();
        let mut token_usage: Option<TokenUsage> = None;
        let mut current_event: Option<String> = None;

        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| CoreError::Llm(format!("stream read error: {e}")))?
        {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            // SSE: `event: <type>` sets the current event type
            if let Some(event_type) = line.strip_prefix("event: ") {
                current_event = Some(event_type.to_string());
                continue;
            }

            // SSE: `data: <json>` dispatches on current event type
            if let Some(data) = line.strip_prefix("data: ") {
                let event = current_event.take();
                match event.as_deref() {
                    Some("response.output_text.delta") => {
                        if let Ok(td) = serde_json::from_str::<TextDelta>(data) {
                            full_response.push_str(&td.delta);
                            if !on_chunk(td.delta) {
                                tracing::debug!("streaming aborted by callback");
                                break;
                            }
                        }
                    }
                    Some("response.output_item.added") => {
                        if let Ok(added) = serde_json::from_str::<OutputItemAdded>(data) {
                            if added.item.r#type == "function_call" {
                                tool_acc.register(
                                    added.output_index,
                                    added.item.call_id.unwrap_or_default(),
                                    added.item.name.unwrap_or_default(),
                                );
                            }
                        }
                    }
                    Some("response.function_call_arguments.delta") => {
                        if let Ok(d) = serde_json::from_str::<FunctionArgsDelta>(data) {
                            tool_acc.append_arguments(d.output_index, &d.delta);
                        }
                    }
                    Some("response.function_call_arguments.done") => {
                        if let Ok(d) = serde_json::from_str::<FunctionArgsDone>(data) {
                            tool_acc.finalize_arguments(d.output_index, &d.arguments);
                        }
                    }
                    Some("response.tool_search_call.searching") => {
                        tracing::info!("tool search initiated");
                    }
                    Some("response.tool_search_call.in_progress") => {
                        // Tool search still running — nothing to do.
                    }
                    Some("response.tool_search_call.completed") => {
                        tracing::info!(data, "tool search completed");
                    }
                    Some("response.failed") => {
                        tracing::warn!("OpenAI response failed: {data}");
                        // Extract a concise error message if possible.
                        let msg = serde_json::from_str::<serde_json::Value>(data)
                            .ok()
                            .and_then(|v| {
                                v.get("response")
                                    .and_then(|r| r.get("error"))
                                    .and_then(|e| e.get("message"))
                                    .and_then(|m| m.as_str())
                                    .map(String::from)
                            })
                            .unwrap_or_else(|| "response.failed".into());
                        return Err(CoreError::Llm(format!("OpenAI server_error: {msg}")));
                    }
                    Some("error") => {
                        tracing::warn!("OpenAI stream error: {data}");
                    }
                    Some("response.completed") => {
                        if let Ok(rc) = serde_json::from_str::<ResponseCompleted>(data) {
                            if let Some(u) = rc.response.usage {
                                token_usage = Some(TokenUsage {
                                    input_tokens: u.input_tokens,
                                    output_tokens: u.output_tokens,
                                    ..Default::default()
                                });
                            }
                        }
                        break;
                    }
                    other => {
                        tracing::debug!("ignoring SSE event: {:?}", other);
                    }
                }
            }
        }

        let tool_calls = tool_acc.into_tool_calls();
        tracing::debug!(
            text_len = full_response.len(),
            tool_call_count = tool_calls.len(),
            "OpenAI response parsed"
        );
        let mut resp = if tool_calls.is_empty() {
            LlmResponse::text(full_response)
        } else {
            LlmResponse::with_tool_calls(full_response, tool_calls)
        };
        if let Some(usage) = token_usage {
            resp = resp.with_usage(usage);
        }
        Ok(resp)
    }
}

/// Curated list of OpenAI models exposed through this connector.
///
/// OpenAI's `/v1/models` endpoint returns a firehose of model IDs
/// (including retired and fine-tune variants) with no capability metadata.
/// Until we ship a resolver that merges the live list with this curated
/// table, we hand-maintain the set that the model picker should offer.
///
/// To extend: append a `ModelInfo` entry here with the right
/// `ModelCapabilities` flags. `reasoning: true` belongs on the o-series
/// (o1, o3, o4) and any GPT-5 variant that exposes reasoning traces.
// TODO(#7): fetch `/v1/models` and merge with this table so newly
// released models surface automatically.
/// True when `model` is one of the curated OpenAI models flagged as
/// reasoning-capable. Used to gate the `reasoning.effort` field on
/// requests — sending it for non-reasoning models (e.g. `gpt-4o`) is a
/// 400 from the API, so we silently drop it with a debug log.
///
/// Matches by exact id AND common prefixes (e.g. `gpt-5-mini-2025…`), so
/// custom pinned versions resolve the same way as their family name.
fn model_supports_reasoning(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // Exact id match against the curated capabilities table.
    if curated_openai_models()
        .iter()
        .any(|c| c.id.eq_ignore_ascii_case(model) && c.capabilities.reasoning)
    {
        return true;
    }
    // Prefix heuristics for pinned/versioned ids not in the curated table.
    //   o-series: `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`, …
    //   GPT-5 reasoning: `gpt-5`, `gpt-5-mini`, `gpt-5.4`, …
    let is_o_series = m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4");
    let is_gpt5 = m.starts_with("gpt-5");
    is_o_series || is_gpt5
}

/// Build a `ReasoningBlock` from the per-turn reasoning hint, honoring
/// the per-model capability gate. Returns `None` for non-reasoning
/// models (debug-logged) and when no effort is requested.
fn reasoning_for(model: &str, reasoning: ReasoningConfig) -> Option<ReasoningBlock> {
    let Some(level) = reasoning.reasoning_effort else {
        return None;
    };
    if !model_supports_reasoning(model) {
        tracing::debug!(
            model,
            requested_effort = ?level,
            "OpenAI reasoning_effort requested but model is not reasoning-capable; dropping field"
        );
        return None;
    }
    Some(ReasoningBlock {
        effort: level.as_openai_effort(),
    })
}

fn curated_openai_models() -> Vec<ModelInfo> {
    let chat_caps = ModelCapabilities {
        reasoning: false,
        vision: true,
        tools: true,
        embedding: false,
    };
    let reasoning_caps = ModelCapabilities {
        reasoning: true,
        vision: true,
        tools: true,
        embedding: false,
    };
    let embedding_caps = ModelCapabilities {
        reasoning: false,
        vision: false,
        tools: false,
        embedding: true,
    };

    vec![
        // --- GPT-5 family ---
        ModelInfo::new("gpt-5")
            .with_display_name("GPT-5")
            .with_context_limit(400_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("gpt-5-mini")
            .with_display_name("GPT-5 Mini")
            .with_context_limit(400_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("gpt-5.4")
            .with_display_name("GPT-5.4")
            .with_context_limit(400_000)
            .with_capabilities(reasoning_caps),
        // --- o-series reasoning models ---
        ModelInfo::new("o4-mini")
            .with_display_name("o4-mini")
            .with_context_limit(200_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("o3")
            .with_display_name("o3")
            .with_context_limit(200_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("o3-mini")
            .with_display_name("o3-mini")
            .with_context_limit(200_000)
            .with_capabilities(reasoning_caps),
        // --- GPT-4.1 / 4o family (fallback general chat) ---
        ModelInfo::new("gpt-4.1")
            .with_display_name("GPT-4.1")
            .with_context_limit(1_000_000)
            .with_capabilities(chat_caps),
        ModelInfo::new("gpt-4o")
            .with_display_name("GPT-4o")
            .with_context_limit(128_000)
            .with_capabilities(chat_caps),
        ModelInfo::new("gpt-4o-mini")
            .with_display_name("GPT-4o mini")
            .with_context_limit(128_000)
            .with_capabilities(chat_caps),
        // --- Embedding models ---
        ModelInfo::new("text-embedding-3-large")
            .with_display_name("Text Embedding 3 Large")
            .with_capabilities(embedding_caps),
        ModelInfo::new("text-embedding-3-small")
            .with_display_name("Text Embedding 3 Small")
            .with_capabilities(embedding_caps),
    ]
}

impl LlmClient for OpenAiClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        let model = current_model_override().unwrap_or_else(|| self.model.clone());
        context_limit_for_model(&model)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        Ok(curated_openai_models())
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let (input, instructions) = convert_messages(&messages);
        let tool_entries: Vec<ToolEntry> = tools
            .iter()
            .map(|t| ToolEntry::Function(FunctionTool::from_definition(t)))
            .collect();

        // Per-turn model override (issue #34): when the daemon-side routing
        // layer has set `MODEL_OVERRIDE`, dispatch the user-chosen model
        // instead of the connector's baked-in `self.model`. The reasoning
        // gating below is keyed on the dispatched model so e.g. a request
        // for `gpt-5` carries `reasoning_effort` even when the connection
        // was built with a non-reasoning default.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        let reasoning_block = reasoning_for(&model, reasoning);

        let request = ResponsesRequest {
            model,
            input,
            instructions,
            stream: true,
            tools: tool_entries,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
            reasoning: reasoning_block,
        };

        let request_json =
            serde_json::to_string(&request).unwrap_or_else(|_| "<serialization error>".into());

        self.send_and_stream(&request_json, &request, on_chunk)
            .await
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.hosted_tool_search
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let (input, instructions) = convert_messages(&messages);

        let mut tool_entries: Vec<ToolEntry> = core_tools
            .iter()
            .map(|t| ToolEntry::Function(FunctionTool::from_definition(t)))
            .collect();

        for ns in namespaces {
            tool_entries.push(ToolEntry::Namespace(NamespaceTool::from_namespace(ns)));
        }

        tool_entries.push(ToolEntry::ToolSearch(ToolSearchSentinel {
            r#type: "tool_search".to_string(),
        }));

        // Per-turn model override (issue #34); see `stream_completion`.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        let reasoning_block = reasoning_for(&model, reasoning);

        let request = ResponsesRequest {
            model,
            input,
            instructions,
            stream: true,
            tools: tool_entries,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
            reasoning: reasoning_block,
        };

        let request_json =
            serde_json::to_string(&request).unwrap_or_else(|_| "<serialization error>".into());

        tracing::info!(
            namespace_count = namespaces.len(),
            core_tool_count = core_tools.len(),
            "using hosted tool search with namespaces"
        );

        self.send_and_stream(&request_json, &request, on_chunk)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::llm::ReasoningLevel;

    // --- convert_messages tests ---

    #[test]
    fn convert_messages_user_only() {
        let msgs = vec![Message::new(Role::User, "hello")];
        let (items, instructions) = convert_messages(&msgs);
        assert!(instructions.is_none());
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0],
            InputItem::Message(InputMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            })
        );
    }

    #[test]
    fn convert_messages_system_becomes_instructions() {
        let msgs = vec![
            Message::new(Role::System, "You are helpful."),
            Message::new(Role::User, "hi"),
        ];
        let (items, instructions) = convert_messages(&msgs);
        assert_eq!(instructions.as_deref(), Some("You are helpful."));
        // System message should NOT appear in items
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn convert_messages_concatenates_system_messages() {
        let msgs = vec![
            Message::new(Role::System, "first"),
            Message::new(Role::System, "second"),
            Message::new(Role::User, "hi"),
        ];
        let (_, instructions) = convert_messages(&msgs);
        assert_eq!(instructions.as_deref(), Some("first\n\nsecond"));
    }

    #[test]
    fn convert_messages_assistant_text() {
        let msgs = vec![Message::new(Role::Assistant, "I can help")];
        let (items, _) = convert_messages(&msgs);
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0],
            InputItem::Message(InputMessage {
                role: "assistant".to_string(),
                content: "I can help".to_string(),
            })
        );
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "read_file", r#"{"path": "/tmp/a"}"#)];
        let msg = Message::assistant_with_tool_calls(calls);
        let (items, _) = convert_messages(&[msg]);
        // Empty content → no InputMessage, just the FunctionCall
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::FunctionCall(fc) => {
                assert_eq!(fc.r#type, "function_call");
                assert_eq!(fc.call_id, "c1");
                assert_eq!(fc.id, "fc_c1");
                assert_eq!(fc.name, "read_file");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn convert_messages_tool_result() {
        let msg = Message::tool_result("c1", "file contents");
        let (items, _) = convert_messages(&[msg]);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::FunctionCallOutput(out) => {
                assert_eq!(out.r#type, "function_call_output");
                assert_eq!(out.call_id, "c1");
                assert_eq!(out.output, "file contents");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn convert_messages_mixed_history() {
        let msgs = vec![
            Message::new(Role::System, "Be concise."),
            Message::new(Role::User, "Read /tmp/a"),
            Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1",
                "read_file",
                r#"{"path":"/tmp/a"}"#,
            )]),
            Message::tool_result("c1", "contents of a"),
            Message::new(Role::Assistant, "Here are the contents."),
        ];
        let (items, instructions) = convert_messages(&msgs);
        assert_eq!(instructions.as_deref(), Some("Be concise."));
        assert_eq!(items.len(), 4); // user, function_call, function_call_output, assistant text
    }

    // --- Tool serialization tests ---

    #[test]
    fn function_tool_serialization_flat_name() {
        let def = ToolDefinition::new("test", "A test tool", serde_json::json!({"type": "object"}));
        let tool = FunctionTool::from_definition(&def);
        let json: serde_json::Value = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "test");
        assert_eq!(json["description"], "A test tool");
        // Name is at top level, NOT nested under "function"
        assert!(json.get("function").is_none());
    }

    #[test]
    fn function_tool_deferred_has_defer_loading() {
        let def = ToolDefinition::new("t", "d", serde_json::json!({}));
        let tool = FunctionTool::from_definition_deferred(&def);
        let json: serde_json::Value = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["defer_loading"], true);
    }

    #[test]
    fn namespace_tool_serialization() {
        let ns = ToolNamespace::new(
            "jira",
            "Jira project tools",
            vec![
                ToolDefinition::new("jira__list", "List issues", serde_json::json!({})),
                ToolDefinition::new("jira__create", "Create issue", serde_json::json!({})),
            ],
        );
        let entry = ToolEntry::Namespace(NamespaceTool::from_namespace(&ns));
        let json: serde_json::Value = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["type"], "namespace");
        assert_eq!(json["name"], "adele_jira");
        assert_eq!(json["description"], "Jira project tools");
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["type"], "function");
        assert!(tools[0]["defer_loading"].as_bool().unwrap());
        assert_eq!(tools[0]["name"], "jira__list");
        // Flat format: no "function" wrapper
        assert!(tools[0].get("function").is_none());
    }

    #[test]
    fn tool_search_sentinel_serialization() {
        let entry = ToolEntry::ToolSearch(ToolSearchSentinel {
            r#type: "tool_search".to_string(),
        });
        let json: serde_json::Value = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["type"], "tool_search");
    }

    // --- Request serialization tests ---

    #[test]
    fn response_request_uses_max_output_tokens() {
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: Some(1024),
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("max_output_tokens"));
        assert!(!json.contains("\"max_tokens\""));
    }

    #[test]
    fn response_request_uses_instructions() {
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: Some("Be helpful.".into()),
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(json["instructions"], "Be helpful.");
    }

    #[test]
    fn response_request_omits_empty_tools() {
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
    }

    #[test]
    fn response_request_with_tools() {
        let def = ToolDefinition::new("test", "desc", serde_json::json!({"type": "object"}));
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![ToolEntry::Function(FunctionTool::from_definition(&def))],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"test\""));
    }

    #[test]
    fn request_with_namespaces_serialization() {
        let core_tool = ToolDefinition::new("mcp_control", "Control MCP", serde_json::json!({}));
        let ns = ToolNamespace::new(
            "kb",
            "Knowledge base tools",
            vec![ToolDefinition::new(
                "kb_write",
                "Write to KB",
                serde_json::json!({}),
            )],
        );

        let mut entries: Vec<ToolEntry> = vec![ToolEntry::Function(FunctionTool::from_definition(
            &core_tool,
        ))];
        entries.push(ToolEntry::Namespace(NamespaceTool::from_namespace(&ns)));
        entries.push(ToolEntry::ToolSearch(ToolSearchSentinel {
            r#type: "tool_search".to_string(),
        }));

        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: entries,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };

        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[1]["type"], "namespace");
        assert_eq!(tools[2]["type"], "tool_search");
    }

    // --- Reasoning / effort tests ----------------------------------------

    #[test]
    fn model_supports_reasoning_curated_ids() {
        assert!(model_supports_reasoning("gpt-5"));
        assert!(model_supports_reasoning("gpt-5-mini"));
        assert!(model_supports_reasoning("gpt-5.4"));
        assert!(model_supports_reasoning("o3"));
        assert!(model_supports_reasoning("o3-mini"));
        assert!(model_supports_reasoning("o4-mini"));
    }

    #[test]
    fn model_supports_reasoning_rejects_chat_models() {
        assert!(!model_supports_reasoning("gpt-4o"));
        assert!(!model_supports_reasoning("gpt-4o-mini"));
        assert!(!model_supports_reasoning("gpt-4.1"));
    }

    #[test]
    fn model_supports_reasoning_prefix_heuristic_versioned_ids() {
        // Pinned versions not in the curated list still resolve correctly.
        assert!(model_supports_reasoning("o1-2024-12-17"));
        assert!(model_supports_reasoning("gpt-5-mini-2025-09-01"));
        assert!(!model_supports_reasoning("gpt-4o-2024-11-20"));
    }

    #[test]
    fn reasoning_for_omits_when_no_effort_requested() {
        assert!(reasoning_for("gpt-5", ReasoningConfig::default()).is_none());
    }

    #[test]
    fn reasoning_for_emits_block_on_reasoning_model() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        let block = reasoning_for("gpt-5.4", cfg).expect("reasoning block expected");
        assert_eq!(block.effort, "high");
    }

    #[test]
    fn reasoning_for_drops_block_on_non_reasoning_model() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        // gpt-4o does not support reasoning — field should be dropped, not sent.
        assert!(reasoning_for("gpt-4o", cfg).is_none());
    }

    #[test]
    fn request_includes_reasoning_for_supported_model() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::Medium);
        let block = reasoning_for("gpt-5", cfg);
        let req = ResponsesRequest {
            model: "gpt-5".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: block,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning"]["effort"], "medium");
    }

    #[test]
    fn request_omits_reasoning_field_when_none() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("reasoning"),
            "reasoning field must be omitted when None; got: {json}"
        );
    }

    // --- Tool accumulator tests ---

    #[test]
    fn response_tool_accumulator_register_and_finalize() {
        let mut acc = ResponseToolAccumulator::default();
        acc.register(0, "call_1".into(), "read_file".into());
        acc.append_arguments(0, r#"{"pa"#);
        acc.append_arguments(0, r#"th": "/tmp"}"#);

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path": "/tmp"}"#);
    }

    #[test]
    fn response_tool_accumulator_finalize_replaces_partial() {
        let mut acc = ResponseToolAccumulator::default();
        acc.register(0, "c1".into(), "tool_a".into());
        acc.append_arguments(0, "partial");
        acc.finalize_arguments(0, r#"{"complete": true}"#);

        let calls = acc.into_tool_calls();
        assert_eq!(calls[0].arguments, r#"{"complete": true}"#);
    }

    #[test]
    fn response_tool_accumulator_multiple_tools() {
        let mut acc = ResponseToolAccumulator::default();
        acc.register(0, "c1".into(), "tool_a".into());
        acc.register(1, "c2".into(), "tool_b".into());
        acc.finalize_arguments(0, "{}");
        acc.finalize_arguments(1, "{}");

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].name, "tool_b");
    }

    #[test]
    fn response_tool_accumulator_sorted_by_index() {
        let mut acc = ResponseToolAccumulator::default();
        // Insert out of order
        acc.register(2, "c3".into(), "tool_c".into());
        acc.register(0, "c1".into(), "tool_a".into());
        acc.register(1, "c2".into(), "tool_b".into());
        acc.finalize_arguments(0, "{}");
        acc.finalize_arguments(1, "{}");
        acc.finalize_arguments(2, "{}");

        let calls = acc.into_tool_calls();
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].name, "tool_b");
        assert_eq!(calls[2].name, "tool_c");
    }

    // --- Client builder test ---

    #[test]
    fn client_builder() {
        let client = OpenAiClient::new("test-key".into())
            .with_model("gpt-3.5-turbo")
            .with_base_url("http://localhost:8080");
        assert_eq!(client.model, "gpt-3.5-turbo");
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn from_env_missing_key() {
        // Ensure the env var is not set for this test
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = OpenAiClient::from_env();
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }

    #[tokio::test]
    async fn list_models_returns_curated_openai_models() {
        let client = OpenAiClient::new("key".into());
        let models = client.list_models().await.unwrap();
        assert!(!models.is_empty());

        // GPT-5 family present with reasoning.
        let gpt5 = models.iter().find(|m| m.id == "gpt-5").unwrap();
        assert!(gpt5.capabilities.reasoning);
        assert!(gpt5.capabilities.tools);
        assert!(!gpt5.capabilities.embedding);
        assert_eq!(gpt5.context_limit, Some(400_000));

        // o-series reasoning flag.
        let o3 = models.iter().find(|m| m.id == "o3").unwrap();
        assert!(o3.capabilities.reasoning);

        // Embedding model has embedding flag, no chat flags.
        let embed = models
            .iter()
            .find(|m| m.id == "text-embedding-3-large")
            .unwrap();
        assert!(embed.capabilities.embedding);
        assert!(!embed.capabilities.reasoning);
        assert!(!embed.capabilities.tools);
        assert!(!embed.capabilities.vision);

        // Non-reasoning chat model example.
        let gpt4o = models.iter().find(|m| m.id == "gpt-4o").unwrap();
        assert!(!gpt4o.capabilities.reasoning);
        assert!(gpt4o.capabilities.tools);
    }

    // --- MODEL_OVERRIDE wiring (issue #34) -------------------------------

    /// Minimal SSE body that lets `send_and_stream` reach `break` on
    /// `response.completed` — empty `response` object is enough.
    const STUB_SSE_BODY: &str =
        "event: response.completed\ndata: {\"response\":{}}\n\n";

    #[tokio::test]
    async fn stream_completion_uses_self_model_when_override_unset() {
        let server = httpmock::MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/responses")
                .body_includes(r#""model":"gpt-5.4""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let _ = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;
        m.assert_calls(1);
    }

    #[tokio::test]
    async fn stream_completion_uses_model_override_when_set() {
        use desktop_assistant_core::ports::llm::with_model_override;

        let server = httpmock::MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/responses")
                .body_includes(r#""model":"gpt-4o-mini""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        with_model_override("gpt-4o-mini".into(), async {
            let _ = client
                .stream_completion(
                    vec![Message::new(Role::User, "hi")],
                    &[],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await;
        })
        .await;
        m.assert_calls(1);
    }

    // --- context_limit_for_model tests ---

    #[test]
    fn context_limit_for_known_models() {
        assert_eq!(context_limit_for_model("gpt-4o"), Some(128_000));
        assert_eq!(context_limit_for_model("gpt-4o-mini"), Some(128_000));
        assert_eq!(
            context_limit_for_model("gpt-4-turbo-2024-04-09"),
            Some(128_000)
        );
        assert_eq!(context_limit_for_model("gpt-4-32k-0613"), Some(32_768));
        assert_eq!(context_limit_for_model("gpt-4-0613"), Some(8_192));
        assert_eq!(
            context_limit_for_model("gpt-3.5-turbo-16k-0613"),
            Some(16_384)
        );
        assert_eq!(context_limit_for_model("gpt-3.5-turbo"), Some(4_096));
        assert_eq!(context_limit_for_model("o1-preview"), Some(200_000));
        assert_eq!(context_limit_for_model("o3-mini"), Some(200_000));
        assert_eq!(context_limit_for_model("o4-mini"), Some(200_000));
    }

    #[test]
    fn context_limit_specific_prefix_wins_over_general() {
        // gpt-4o is checked before the bare gpt-4 fallback, so we get
        // 128k rather than 8k.
        assert_eq!(context_limit_for_model("gpt-4o-2024-08-06"), Some(128_000));
        // Likewise gpt-4-turbo is 128k, not 8k.
        assert_eq!(context_limit_for_model("gpt-4-turbo"), Some(128_000));
        // Likewise gpt-4-32k is 32k, not 8k.
        assert_eq!(context_limit_for_model("gpt-4-32k"), Some(32_768));
    }

    #[test]
    fn context_limit_for_gpt5_returns_none() {
        // GPT-5 family is intentionally excluded — the daemon's universal
        // fallback handles it until we curate per-tier values.
        assert_eq!(context_limit_for_model("gpt-5"), None);
        assert_eq!(context_limit_for_model("gpt-5-mini"), None);
        assert_eq!(context_limit_for_model("gpt-5.4"), None);
    }

    #[test]
    fn context_limit_for_unknown_returns_none() {
        assert_eq!(context_limit_for_model("davinci"), None);
        assert_eq!(context_limit_for_model("text-embedding-3-large"), None);
        assert_eq!(context_limit_for_model("totally-fake-model"), None);
    }

    #[test]
    fn max_context_tokens_uses_configured_model() {
        let client = OpenAiClient::new("k".into()).with_model("gpt-4o");
        assert_eq!(client.max_context_tokens(), Some(128_000));
    }

    #[tokio::test]
    async fn max_context_tokens_consults_model_override() {
        use desktop_assistant_core::ports::llm::with_model_override;

        let client = OpenAiClient::new("k".into()).with_model("totally-fake-model");
        assert_eq!(client.max_context_tokens(), None);
        let observed =
            with_model_override("gpt-4o-mini".into(), async { client.max_context_tokens() })
                .await;
        assert_eq!(observed, Some(128_000));
    }
}
