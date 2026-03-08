use std::collections::HashMap;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse, TokenUsage};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio_stream::StreamExt;

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
            hosted_tool_search: true,
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
                        return Err(CoreError::Llm(format!(
                            "OpenAI server_error: {msg}"
                        )));
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

impl LlmClient for OpenAiClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let (input, instructions) = convert_messages(&messages);
        let tool_entries: Vec<ToolEntry> = tools
            .iter()
            .map(|t| ToolEntry::Function(FunctionTool::from_definition(t)))
            .collect();

        let request = ResponsesRequest {
            model: self.model.clone(),
            input,
            instructions,
            stream: true,
            tools: tool_entries,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
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

        let request = ResponsesRequest {
            model: self.model.clone(),
            input,
            instructions,
            stream: true,
            tools: tool_entries,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
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
        };

        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[1]["type"], "namespace");
        assert_eq!(tools[2]["type"], "tool_search");
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
}
