use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse, TokenUsage};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio_stream::StreamExt;

/// Anthropic Messages API client that streams completions via SSE.
pub struct AnthropicClient {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    temperature: Option<f64>,
    top_p: Option<f64>,
    hosted_tool_search: bool,
}

impl AnthropicClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("claude-sonnet-4-6-20260227")
    }

    pub fn get_default_base_url() -> Option<&'static str> {
        Some("https://api.anthropic.com")
    }

    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: Self::get_default_model().unwrap_or_default().to_string(),
            base_url: Self::get_default_base_url().unwrap_or_default().to_string(),
            max_tokens: 8192,
            temperature: None,
            top_p: None,
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

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_max_tokens_override(mut self, max_tokens: Option<u32>) -> Self {
        if let Some(mt) = max_tokens {
            self.max_tokens = mt;
        }
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

    pub fn with_hosted_tool_search(mut self, enabled: bool) -> Self {
        self.hosted_tool_search = enabled;
        self
    }

    /// Create from environment variables.
    /// Reads `ANTHROPIC_API_KEY` for the API key.
    /// Optionally reads `ANTHROPIC_MODEL` (defaults to claude-sonnet-4-6-20260227)
    /// and `ANTHROPIC_BASE_URL` (defaults to https://api.anthropic.com).
    pub fn from_env() -> Result<Self, CoreError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| CoreError::Llm("ANTHROPIC_API_KEY environment variable not set".into()))?;
        let mut client = Self::new(api_key);
        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            client.model = model;
        }
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            client.base_url = url;
        }
        Ok(client)
    }
}

// --- Request types ---

#[derive(Serialize, Clone)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
}

impl CacheControl {
    fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral",
        }
    }
}

#[derive(Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
    cache_control: CacheControl,
}

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemBlock>,
    messages: Vec<AnthropicMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

#[derive(Serialize, Debug, Clone)]
struct AnthropicMessage {
    role: String,
    content: Vec<ContentBlock>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl From<&ToolDefinition> for AnthropicTool {
    fn from(def: &ToolDefinition) -> Self {
        AnthropicTool {
            name: def.name.clone(),
            description: def.description.clone(),
            input_schema: def.parameters.clone(),
        }
    }
}

/// A tool with `defer_loading: true` for Anthropic's hosted tool search.
#[derive(Serialize)]
struct AnthropicDeferredTool {
    r#type: String,
    name: String,
    description: String,
    input_schema: serde_json::Value,
    defer_loading: bool,
}

impl AnthropicDeferredTool {
    fn from_definition(def: &ToolDefinition) -> Self {
        Self {
            r#type: "custom".to_string(),
            name: def.name.clone(),
            description: def.description.clone(),
            input_schema: def.parameters.clone(),
            defer_loading: true,
        }
    }
}

/// The Anthropic tool search sentinel tool.
#[derive(Serialize)]
struct AnthropicToolSearchTool {
    r#type: String,
    name: String,
}

/// Untagged enum for the tools array with hosted tool search.
#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicToolEntry {
    Regular(AnthropicTool),
    Deferred(AnthropicDeferredTool),
    ToolSearch(AnthropicToolSearchTool),
}

/// Request variant using `AnthropicToolEntry` for tool search support.
#[derive(Serialize)]
struct MessagesRequestWithToolSearch {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemBlock>,
    messages: Vec<AnthropicMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicToolEntry>,
}

/// Convert domain messages into Anthropic API messages, extracting the system prompt.
///
/// The system prompt is returned as a `Vec<SystemBlock>` with an ephemeral cache
/// breakpoint so the Anthropic API caches it across turns.
fn convert_messages(messages: &[Message]) -> (Vec<SystemBlock>, Vec<AnthropicMessage>) {
    let mut system_blocks: Vec<SystemBlock> = Vec::new();
    let mut api_messages: Vec<AnthropicMessage> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                system_blocks.push(SystemBlock {
                    block_type: "text",
                    text: msg.content.clone(),
                    cache_control: CacheControl::ephemeral(),
                });
            }
            Role::User => {
                api_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![ContentBlock::Text {
                        text: msg.content.clone(),
                    }],
                });
            }
            Role::Assistant => {
                let mut content: Vec<ContentBlock> = Vec::new();
                if !msg.content.is_empty() {
                    content.push(ContentBlock::Text {
                        text: msg.content.clone(),
                    });
                }
                for tc in &msg.tool_calls {
                    let input: serde_json::Value = serde_json::from_str(&tc.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    content.push(ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input,
                    });
                }
                if content.is_empty() {
                    content.push(ContentBlock::Text {
                        text: String::new(),
                    });
                }
                api_messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content,
                });
            }
            Role::Tool => {
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                api_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id,
                        content: msg.content.clone(),
                    }],
                });
            }
        }
    }

    (system_blocks, api_messages)
}

// --- SSE response types ---

#[derive(Deserialize, Debug, Default)]
struct SseUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct SseEvent {
    #[serde(rename = "type")]
    event_type: String,
    index: Option<usize>,
    content_block: Option<SseContentBlock>,
    #[serde(default, deserialize_with = "deserialize_optional_delta")]
    delta: Option<SseDelta>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

/// Deserialize `delta` permissively: returns `None` when the object shape
/// doesn't match `SseDelta` (e.g. `message_delta` events whose delta lacks a
/// `type` tag).
fn deserialize_optional_delta<'de, D>(deserializer: D) -> Result<Option<SseDelta>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match value {
        Some(v) => Ok(serde_json::from_value(v).ok()),
        None => Ok(None),
    }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum SseContentBlock {
    #[serde(rename = "text")]
    Text {
        #[allow(dead_code)]
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum SseDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

/// Accumulator for building tool calls from streaming content blocks.
#[derive(Default)]
struct ToolCallAccumulator {
    entries: Vec<ToolCallEntry>,
}

#[derive(Default)]
struct ToolCallEntry {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn start_tool_use(&mut self, index: usize, id: String, name: String) {
        while self.entries.len() <= index {
            self.entries.push(ToolCallEntry::default());
        }
        let entry = &mut self.entries[index];
        entry.id = id;
        entry.name = name;
    }

    fn append_json(&mut self, index: usize, partial_json: &str) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.arguments.push_str(partial_json);
        }
    }

    fn into_tool_calls(self) -> Vec<ToolCall> {
        self.entries
            .into_iter()
            .map(|e| ToolCall::new(e.id, e.name, e.arguments))
            .collect()
    }
}

impl AnthropicClient {
    /// Send a request and parse the SSE stream into an LlmResponse.
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

        let url = format!("{}/v1/messages", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .header(
                "anthropic-beta",
                "interleaved-thinking-2025-05-14,tool-search-2025-04-15",
            )
            .json(request_body)
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("HTTP request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "Anthropic API error (HTTP {status}): {body}"
            )));
        }

        let byte_stream = response.bytes_stream();
        let mapped_stream = byte_stream.map(|result: Result<bytes::Bytes, reqwest::Error>| {
            result.map_err(std::io::Error::other)
        });
        let stream_reader = tokio_util::io::StreamReader::new(mapped_stream);
        let mut lines = tokio::io::BufReader::new(stream_reader).lines();

        let mut full_response = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut accumulated_usage = SseUsage::default();
        let mut tool_block_count: usize = 0;
        let mut block_to_tool: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| CoreError::Llm(format!("stream read error: {e}")))?
        {
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }

                match serde_json::from_str::<SseEvent>(data) {
                    Ok(event) => match event.event_type.as_str() {
                        "content_block_start" => {
                            if let (Some(index), Some(block)) = (event.index, &event.content_block)
                            {
                                match block {
                                    SseContentBlock::ToolUse { id, name } => {
                                        let tool_idx = tool_block_count;
                                        tool_block_count += 1;
                                        block_to_tool.insert(index, tool_idx);
                                        tool_acc.start_tool_use(tool_idx, id.clone(), name.clone());
                                    }
                                    SseContentBlock::Text { .. } => {}
                                }
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = &event.delta {
                                match delta {
                                    SseDelta::TextDelta { text } => {
                                        full_response.push_str(text);
                                        if !on_chunk(text.clone()) {
                                            tracing::debug!("streaming aborted by callback");
                                            return Ok(LlmResponse::text(full_response));
                                        }
                                    }
                                    SseDelta::InputJsonDelta { partial_json } => {
                                        if let Some(index) = event.index
                                            && let Some(&tool_idx) = block_to_tool.get(&index)
                                        {
                                            tool_acc.append_json(tool_idx, partial_json);
                                        }
                                    }
                                }
                            }
                        }
                        "message_stop" => {
                            break;
                        }
                        "message_start" | "message_delta" => {
                            if let Some(u) = &event.usage {
                                accumulate_usage(&mut accumulated_usage, u);
                            }
                        }
                        _ => {
                            // content_block_stop, ping — ignore
                        }
                    },
                    Err(e) => {
                        tracing::warn!("failed to parse SSE event: {e}, data: {data}");
                    }
                }
            }
        }

        let tool_calls = tool_acc.into_tool_calls();
        let usage = to_token_usage(&accumulated_usage);
        let resp = if tool_calls.is_empty() {
            LlmResponse::text(full_response)
        } else {
            LlmResponse::with_tool_calls(full_response, tool_calls)
        };
        Ok(resp.with_usage(usage))
    }
}

impl LlmClient for AnthropicClient {
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
        let api_tools: Vec<AnthropicTool> = tools.iter().map(AnthropicTool::from).collect();
        let (system, api_messages) = convert_messages(&messages);

        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            system,
            messages: api_messages,
            stream: true,
            tools: api_tools,
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
        let (system, api_messages) = convert_messages(&messages);

        // Core tools are sent as regular (non-deferred) tools
        let mut tool_entries: Vec<AnthropicToolEntry> = core_tools
            .iter()
            .map(|t| AnthropicToolEntry::Regular(AnthropicTool::from(t)))
            .collect();

        // Namespace tools are sent as deferred
        for ns in namespaces {
            for tool in &ns.tools {
                tool_entries.push(AnthropicToolEntry::Deferred(
                    AnthropicDeferredTool::from_definition(tool),
                ));
            }
        }

        // Add the tool search sentinel
        tool_entries.push(AnthropicToolEntry::ToolSearch(AnthropicToolSearchTool {
            r#type: "tool_search_tool_regex_20251119".to_string(),
            name: "tool_search_tool_regex".to_string(),
        }));

        let request = MessagesRequestWithToolSearch {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            system,
            messages: api_messages,
            stream: true,
            tools: tool_entries,
        };

        let request_json =
            serde_json::to_string(&request).unwrap_or_else(|_| "<serialization error>".into());

        tracing::info!(
            namespace_count = namespaces.len(),
            core_tool_count = core_tools.len(),
            "using Anthropic hosted tool search with deferred tools"
        );

        self.send_and_stream(&request_json, &request, on_chunk)
            .await
    }
}

fn accumulate_usage(acc: &mut SseUsage, new: &SseUsage) {
    fn add(a: &mut Option<u64>, b: Option<u64>) {
        if let Some(v) = b {
            *a = Some(a.unwrap_or(0) + v);
        }
    }
    add(&mut acc.input_tokens, new.input_tokens);
    add(&mut acc.output_tokens, new.output_tokens);
    add(
        &mut acc.cache_creation_input_tokens,
        new.cache_creation_input_tokens,
    );
    add(
        &mut acc.cache_read_input_tokens,
        new.cache_read_input_tokens,
    );
}

fn to_token_usage(sse: &SseUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: sse.input_tokens,
        output_tokens: sse.output_tokens,
        cache_creation_input_tokens: sse.cache_creation_input_tokens,
        cache_read_input_tokens: sse.cache_read_input_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_builder() {
        let client = AnthropicClient::new("test-key".into())
            .with_model("claude-haiku-3-5")
            .with_base_url("http://localhost:8080")
            .with_max_tokens(4096);
        assert_eq!(client.model, "claude-haiku-3-5");
        assert_eq!(client.base_url, "http://localhost:8080");
        assert_eq!(client.max_tokens, 4096);
        assert_eq!(client.api_key, "test-key");
    }

    #[test]
    fn client_defaults() {
        let client = AnthropicClient::new("key".into());
        assert_eq!(client.model, "claude-sonnet-4-6-20260227");
        assert_eq!(client.base_url, "https://api.anthropic.com");
        assert_eq!(client.max_tokens, 8192);
    }

    #[test]
    fn convert_user_message() {
        let messages = vec![Message::new(Role::User, "hello")];
        let (system, api_msgs) = convert_messages(&messages);
        assert!(system.is_empty());
        assert_eq!(api_msgs.len(), 1);
        assert_eq!(api_msgs[0].role, "user");
        match &api_msgs[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn convert_system_message_extracted() {
        let messages = vec![
            Message::new(Role::System, "you are helpful"),
            Message::new(Role::User, "hi"),
        ];
        let (system, api_msgs) = convert_messages(&messages);
        assert_eq!(system.len(), 1);
        assert_eq!(system[0].text, "you are helpful");
        assert_eq!(system[0].cache_control.cache_type, "ephemeral");
        assert_eq!(api_msgs.len(), 1); // system not in messages array
        assert_eq!(api_msgs[0].role, "user");
    }

    #[test]
    fn convert_assistant_message() {
        let messages = vec![Message::new(Role::Assistant, "sure")];
        let (_, api_msgs) = convert_messages(&messages);
        assert_eq!(api_msgs[0].role, "assistant");
        match &api_msgs[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "sure"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn convert_assistant_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "read_file", r#"{"path": "/tmp/a"}"#)];
        let msg = Message::assistant_with_tool_calls(calls);
        let (_, api_msgs) = convert_messages(&[msg]);
        assert_eq!(api_msgs[0].role, "assistant");
        assert_eq!(api_msgs[0].content.len(), 1); // no text, just tool_use
        match &api_msgs[0].content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "/tmp/a");
            }
            _ => panic!("expected tool_use block"),
        }
    }

    #[test]
    fn convert_tool_result_message() {
        let msg = Message::tool_result("call-1", "file contents");
        let (_, api_msgs) = convert_messages(&[msg]);
        assert_eq!(api_msgs[0].role, "user");
        match &api_msgs[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "call-1");
                assert_eq!(content, "file contents");
            }
            _ => panic!("expected tool_result block"),
        }
    }

    #[test]
    fn tool_definition_conversion() {
        let def = ToolDefinition::new("test", "A test tool", serde_json::json!({"type": "object"}));
        let tool = AnthropicTool::from(&def);
        assert_eq!(tool.name, "test");
        assert_eq!(tool.description, "A test tool");
        assert_eq!(tool.input_schema, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn parse_content_block_start_tool_use() {
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"read_file"}}"#;
        let event: SseEvent = serde_json::from_str(data).unwrap();
        assert_eq!(event.event_type, "content_block_start");
        assert_eq!(event.index, Some(1));
        match event.content_block.unwrap() {
            SseContentBlock::ToolUse { id, name } => {
                assert_eq!(id, "toolu_abc");
                assert_eq!(name, "read_file");
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn parse_content_block_start_text() {
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let event: SseEvent = serde_json::from_str(data).unwrap();
        assert_eq!(event.event_type, "content_block_start");
        match event.content_block.unwrap() {
            SseContentBlock::Text { text } => assert_eq!(text, ""),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_content_block_delta_text() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: SseEvent = serde_json::from_str(data).unwrap();
        assert_eq!(event.event_type, "content_block_delta");
        match event.delta.unwrap() {
            SseDelta::TextDelta { text } => assert_eq!(text, "Hello"),
            _ => panic!("expected text_delta"),
        }
    }

    #[test]
    fn parse_content_block_delta_input_json() {
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}"#;
        let event: SseEvent = serde_json::from_str(data).unwrap();
        assert_eq!(event.event_type, "content_block_delta");
        match event.delta.unwrap() {
            SseDelta::InputJsonDelta { partial_json } => assert_eq!(partial_json, "{\"pa"),
            _ => panic!("expected input_json_delta"),
        }
    }

    #[test]
    fn parse_message_delta_event_succeeds() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":87,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":8}}"#;
        let event: SseEvent = serde_json::from_str(data).unwrap();
        assert_eq!(event.event_type, "message_delta");
        assert!(event.delta.is_none()); // delta shape doesn't match SseDelta
        let usage = event.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(87));
        assert_eq!(usage.output_tokens, Some(8));
        assert_eq!(usage.cache_creation_input_tokens, Some(0));
        assert_eq!(usage.cache_read_input_tokens, Some(0));
    }

    #[test]
    fn accumulate_usage_sums_values() {
        let mut acc = SseUsage::default();
        let start = SseUsage {
            input_tokens: Some(100),
            output_tokens: None,
            cache_creation_input_tokens: Some(10),
            cache_read_input_tokens: None,
        };
        let delta = SseUsage {
            input_tokens: None,
            output_tokens: Some(50),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(20),
        };
        accumulate_usage(&mut acc, &start);
        accumulate_usage(&mut acc, &delta);
        let usage = to_token_usage(&acc);
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(20));
    }

    #[test]
    fn tool_call_accumulator_builds_from_blocks() {
        let mut acc = ToolCallAccumulator::default();

        acc.start_tool_use(0, "toolu_1".into(), "read_file".into());
        acc.append_json(0, "{\"pa");
        acc.append_json(0, "th\": \"/tmp\"}");

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path": "/tmp"}"#);
    }

    #[test]
    fn tool_call_accumulator_multiple_tools() {
        let mut acc = ToolCallAccumulator::default();

        acc.start_tool_use(0, "t1".into(), "tool_a".into());
        acc.append_json(0, "{}");

        acc.start_tool_use(1, "t2".into(), "tool_b".into());
        acc.append_json(1, "{}");

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].name, "tool_b");
    }

    #[test]
    fn request_without_tools_omits_field() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6-20260227".into(),
            max_tokens: 8192,
            temperature: None,
            top_p: None,
            system: vec![],
            messages: vec![],
            stream: true,
            tools: vec![],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
    }

    #[test]
    fn request_with_tools_includes_field() {
        let def = ToolDefinition::new("test", "desc", serde_json::json!({"type": "object"}));
        let req = MessagesRequest {
            model: "claude-sonnet-4-6-20260227".into(),
            max_tokens: 8192,
            temperature: None,
            top_p: None,
            system: vec![SystemBlock {
                block_type: "text",
                text: "system prompt".into(),
                cache_control: CacheControl::ephemeral(),
            }],
            messages: vec![],
            stream: true,
            tools: vec![AnthropicTool::from(&def)],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"input_schema\""));
        assert!(json.contains("\"test\""));
    }

    #[test]
    fn request_with_system_includes_field() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6-20260227".into(),
            max_tokens: 8192,
            temperature: None,
            top_p: None,
            system: vec![SystemBlock {
                block_type: "text",
                text: "be helpful".into(),
                cache_control: CacheControl::ephemeral(),
            }],
            messages: vec![],
            stream: true,
            tools: vec![],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"system\""));
        assert!(json.contains("\"be helpful\""));
        assert!(json.contains("\"cache_control\""));
        assert!(json.contains("\"ephemeral\""));
    }

    #[test]
    fn from_env_missing_key() {
        // Ensure the env var is not set for this test
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
        let result = AnthropicClient::from_env();
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }

    #[test]
    fn anthropic_deferred_tool_serialization() {
        let def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let deferred = AnthropicDeferredTool::from_definition(&def);
        let json: serde_json::Value = serde_json::to_value(&deferred).unwrap();
        assert_eq!(json["name"], "read_file");
        assert_eq!(json["description"], "Read a file");
        assert!(json["defer_loading"].as_bool().unwrap());
        assert_eq!(json["type"], "custom");
    }

    #[test]
    fn anthropic_tool_search_sentinel_serialization() {
        let sentinel = AnthropicToolSearchTool {
            r#type: "tool_search_tool_regex_20251119".to_string(),
            name: "tool_search_tool_regex".to_string(),
        };
        let json: serde_json::Value = serde_json::to_value(&sentinel).unwrap();
        assert_eq!(json["type"], "tool_search_tool_regex_20251119");
        assert_eq!(json["name"], "tool_search_tool_regex");
    }

    #[test]
    fn anthropic_tool_entry_mixed_serialization() {
        let core = AnthropicToolEntry::Regular(AnthropicTool {
            name: "mcp_control".into(),
            description: "Control MCP".into(),
            input_schema: serde_json::json!({}),
        });
        let deferred = AnthropicToolEntry::Deferred(AnthropicDeferredTool {
            r#type: "custom".into(),
            name: "jira__create".into(),
            description: "Create Jira issue".into(),
            input_schema: serde_json::json!({}),
            defer_loading: true,
        });
        let sentinel = AnthropicToolEntry::ToolSearch(AnthropicToolSearchTool {
            r#type: "tool_search_tool_regex_20251119".into(),
            name: "tool_search_tool_regex".into(),
        });

        let entries = vec![core, deferred, sentinel];
        let json: serde_json::Value = serde_json::to_value(&entries).unwrap();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        // Core tool: no defer_loading
        assert!(arr[0].get("defer_loading").is_none());
        assert_eq!(arr[0]["name"], "mcp_control");
        // Deferred tool: has defer_loading
        assert!(arr[1]["defer_loading"].as_bool().unwrap());
        // Sentinel: has type field
        assert_eq!(arr[2]["type"], "tool_search_tool_regex_20251119");
    }

    #[test]
    fn convert_multiple_system_messages_all_preserved() {
        let messages = vec![
            Message::new(Role::System, "main instruction"),
            Message::new(Role::User, "hi"),
            Message::new(Role::System, "context summary"),
            Message::new(Role::System, "message summary"),
        ];
        let (system, api_msgs) = convert_messages(&messages);
        assert_eq!(system.len(), 3);
        assert_eq!(system[0].text, "main instruction");
        assert_eq!(system[1].text, "context summary");
        assert_eq!(system[2].text, "message summary");
        for block in &system {
            assert_eq!(block.cache_control.cache_type, "ephemeral");
        }
        assert_eq!(api_msgs.len(), 1);
        assert_eq!(api_msgs[0].role, "user");
    }

    #[test]
    fn request_without_system_omits_field() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-6-20260227".into(),
            max_tokens: 8192,
            temperature: None,
            top_p: None,
            system: vec![],
            messages: vec![],
            stream: true,
            tools: vec![],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("system"));
    }
}
