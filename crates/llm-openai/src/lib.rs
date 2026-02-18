use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio_stream::StreamExt;

/// OpenAI-compatible LLM client that streams completions via SSE.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAiClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("gpt-5.2")
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

    /// Create from environment variables.
    /// Reads `OPENAI_API_KEY` for the API key.
    /// Optionally reads `OPENAI_MODEL` (defaults to gpt-5.2)
    /// and `OPENAI_BASE_URL` (defaults to https://api.openai.com/v1).
    /// Generate embeddings for a batch of texts.
    ///
    /// Sends a `POST {base_url}/embeddings` request and returns one vector per input.
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

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
}

/// OpenAI tool definition wrapper.
#[derive(Serialize)]
struct ChatTool {
    r#type: String,
    function: ChatFunction,
}

#[derive(Serialize)]
struct ChatFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&ToolDefinition> for ChatTool {
    fn from(def: &ToolDefinition) -> Self {
        ChatTool {
            r#type: "function".to_string(),
            function: ChatFunction {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.parameters.clone(),
            },
        }
    }
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatMessageToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Tool call as included in an assistant message being sent to the API.
#[derive(Serialize)]
struct ChatMessageToolCall {
    id: String,
    r#type: String,
    function: ChatMessageFunction,
}

#[derive(Serialize)]
struct ChatMessageFunction {
    name: String,
    arguments: String,
}

impl From<&Message> for ChatMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        };

        let tool_calls = if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| ChatMessageToolCall {
                        id: tc.id.clone(),
                        r#type: "function".to_string(),
                        function: ChatMessageFunction {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect(),
            )
        };

        let content = if msg.content.is_empty() && tool_calls.is_some() {
            None
        } else {
            Some(msg.content.clone())
        };

        ChatMessage {
            role: role.to_string(),
            content,
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

// --- Embedding response types ---

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// --- Response deserialization types ---

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Delta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
    tool_calls: Option<Vec<DeltaToolCall>>,
}

/// A tool call delta from the streaming response.
#[derive(Deserialize)]
struct DeltaToolCall {
    index: usize,
    id: Option<String>,
    function: Option<DeltaFunction>,
}

#[derive(Deserialize)]
struct DeltaFunction {
    name: Option<String>,
    arguments: Option<String>,
}

/// Accumulator for building tool calls from streaming deltas.
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
    fn apply_delta(&mut self, delta: &DeltaToolCall) {
        // Grow entries vector if needed
        while self.entries.len() <= delta.index {
            self.entries.push(ToolCallEntry::default());
        }

        let entry = &mut self.entries[delta.index];

        if let Some(id) = &delta.id {
            entry.id = id.clone();
        }
        if let Some(func) = &delta.function {
            if let Some(name) = &func.name {
                entry.name = name.clone();
            }
            if let Some(args) = &func.arguments {
                entry.arguments.push_str(args);
            }
        }
    }

    fn into_tool_calls(self) -> Vec<ToolCall> {
        self.entries
            .into_iter()
            .map(|e| ToolCall::new(e.id, e.name, e.arguments))
            .collect()
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
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let chat_tools: Vec<ChatTool> = tools.iter().map(ChatTool::from).collect();

        let request = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(ChatMessage::from).collect(),
            stream: true,
            tools: chat_tools,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
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
        let mut tool_acc = ToolCallAccumulator::default();

        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| CoreError::Llm(format!("stream read error: {e}")))?
        {
            let line: String = line.trim().to_string();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }

                match serde_json::from_str::<ChatChunk>(data) {
                    Ok(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            // Handle text content
                            if let Some(content) = &choice.delta.content {
                                full_response.push_str(content);
                                if !on_chunk(content.clone()) {
                                    tracing::debug!("streaming aborted by callback");
                                    break;
                                }
                            }

                            // Handle tool call deltas
                            if let Some(tool_calls) = &choice.delta.tool_calls {
                                for tc_delta in tool_calls {
                                    tool_acc.apply_delta(tc_delta);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("failed to parse SSE chunk: {e}, data: {data}");
                    }
                }
            }
        }

        let tool_calls = tool_acc.into_tool_calls();
        if tool_calls.is_empty() {
            Ok(LlmResponse::text(full_response))
        } else {
            Ok(LlmResponse::with_tool_calls(full_response, tool_calls))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_from_domain_message() {
        let msg = Message::new(Role::User, "hello");
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "user");
        assert_eq!(chat_msg.content.as_deref(), Some("hello"));
        assert!(chat_msg.tool_calls.is_none());
        assert!(chat_msg.tool_call_id.is_none());
    }

    #[test]
    fn chat_message_from_assistant() {
        let msg = Message::new(Role::Assistant, "hi");
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "assistant");
    }

    #[test]
    fn chat_message_from_system() {
        let msg = Message::new(Role::System, "instructions");
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "system");
    }

    #[test]
    fn chat_message_from_tool_result() {
        let msg = Message::tool_result("call-1", "file contents");
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "tool");
        assert_eq!(chat_msg.content.as_deref(), Some("file contents"));
        assert_eq!(chat_msg.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn chat_message_from_assistant_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "read_file", r#"{"path": "/tmp/a"}"#)];
        let msg = Message::assistant_with_tool_calls(calls);
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "assistant");
        assert!(chat_msg.content.is_none()); // empty content omitted
        let tc = chat_msg.tool_calls.unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "c1");
        assert_eq!(tc[0].function.name, "read_file");
    }

    #[test]
    fn chat_tool_from_tool_definition() {
        let def = ToolDefinition::new("test", "A test tool", serde_json::json!({"type": "object"}));
        let chat_tool = ChatTool::from(&def);
        assert_eq!(chat_tool.r#type, "function");
        assert_eq!(chat_tool.function.name, "test");
        assert_eq!(chat_tool.function.description, "A test tool");
    }

    #[test]
    fn client_builder() {
        let client = OpenAiClient::new("test-key".into())
            .with_model("gpt-3.5-turbo")
            .with_base_url("http://localhost:8080");
        assert_eq!(client.model, "gpt-3.5-turbo");
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn parse_sse_chunk() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_ref().unwrap(), "Hello");
    }

    #[test]
    fn parse_sse_chunk_no_content() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":null}]}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn parse_sse_chunk_with_tool_calls() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"read_file","arguments":"{\"pa"}}]},"finish_reason":null}]}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        let tc = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].index, 0);
        assert_eq!(tc[0].id.as_deref(), Some("call_abc"));
        assert_eq!(
            tc[0].function.as_ref().unwrap().name.as_deref(),
            Some("read_file")
        );
    }

    #[test]
    fn tool_call_accumulator_builds_from_deltas() {
        let mut acc = ToolCallAccumulator::default();

        // First delta: id and name
        acc.apply_delta(&DeltaToolCall {
            index: 0,
            id: Some("call_1".into()),
            function: Some(DeltaFunction {
                name: Some("read_file".into()),
                arguments: Some("{\"pa".into()),
            }),
        });

        // Second delta: more arguments
        acc.apply_delta(&DeltaToolCall {
            index: 0,
            id: None,
            function: Some(DeltaFunction {
                name: None,
                arguments: Some("th\": \"/tmp\"}".into()),
            }),
        });

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path": "/tmp"}"#);
    }

    #[test]
    fn tool_call_accumulator_multiple_tools() {
        let mut acc = ToolCallAccumulator::default();

        acc.apply_delta(&DeltaToolCall {
            index: 0,
            id: Some("c1".into()),
            function: Some(DeltaFunction {
                name: Some("tool_a".into()),
                arguments: Some("{}".into()),
            }),
        });

        acc.apply_delta(&DeltaToolCall {
            index: 1,
            id: Some("c2".into()),
            function: Some(DeltaFunction {
                name: Some("tool_b".into()),
                arguments: Some("{}".into()),
            }),
        });

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].name, "tool_b");
    }

    #[test]
    fn request_without_tools_omits_field() {
        let req = ChatRequest {
            model: "gpt-5.2".into(),
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
        let req = ChatRequest {
            model: "gpt-5.2".into(),
            messages: vec![],
            stream: true,
            tools: vec![ChatTool::from(&def)],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"function\""));
        assert!(json.contains("\"test\""));
    }

    #[test]
    fn from_env_missing_key() {
        // Ensure the env var is not set for this test
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = OpenAiClient::from_env();
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }
}
