use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

/// Ollama LLM client that streams completions via the native `/api/chat` endpoint.
///
/// Uses NDJSON streaming (one JSON object per line) and Ollama's native tool
/// calling format. No authentication is required.
pub struct OllamaClient {
    client: Client,
    model: String,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into(),
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
}

// --- Request types ---

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatMessageToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct ChatMessageToolCall {
    function: ChatMessageFunction,
}

#[derive(Serialize)]
struct ChatMessageFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct ChatTool {
    r#type: String,
    function: ChatToolFunction,
}

#[derive(Serialize)]
struct ChatToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&ToolDefinition> for ChatTool {
    fn from(def: &ToolDefinition) -> Self {
        ChatTool {
            r#type: "function".to_string(),
            function: ChatToolFunction {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.parameters.clone(),
            },
        }
    }
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
                    .map(|tc| {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Object(
                                serde_json::Map::new(),
                            ));
                        ChatMessageToolCall {
                            function: ChatMessageFunction {
                                name: tc.name.clone(),
                                arguments,
                            },
                        }
                    })
                    .collect(),
            )
        };

        ChatMessage {
            role: role.to_string(),
            content: msg.content.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

// --- Response types ---

#[derive(Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    done: bool,
}

#[derive(Deserialize)]
struct ChunkMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    function: ResponseFunction,
}

#[derive(Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: serde_json::Value,
}

impl LlmClient for OllamaClient {
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

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(&url)
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
                "Ollama API error (HTTP {status}): {body}"
            )));
        }

        // NDJSON streaming: each line is a complete JSON object
        let mut stream = response.bytes_stream();
        let mut full_response = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| CoreError::Llm(format!("stream read error: {e}")))?;
            buffer.push_str(
                &String::from_utf8_lossy(&bytes),
            );

            // Process complete lines from the buffer
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<ChatChunk>(&line) {
                    Ok(chunk) => {
                        if let Some(message) = &chunk.message {
                            if let Some(content) = &message.content {
                                if !content.is_empty() {
                                    full_response.push_str(content);
                                    if !on_chunk(content.clone()) {
                                        tracing::debug!("streaming aborted by callback");
                                        return Ok(build_response(full_response, tool_calls));
                                    }
                                }
                            }

                            if let Some(tcs) = &message.tool_calls {
                                for (i, tc) in tcs.iter().enumerate() {
                                    let id = format!("ollama_call_{}", tool_calls.len() + i);
                                    let arguments = serde_json::to_string(&tc.function.arguments)
                                        .unwrap_or_else(|_| "{}".to_string());
                                    tool_calls.push(ToolCall::new(
                                        id,
                                        tc.function.name.clone(),
                                        arguments,
                                    ));
                                }
                            }
                        }

                        if chunk.done {
                            return Ok(build_response(full_response, tool_calls));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("failed to parse NDJSON chunk: {e}, line: {line}");
                    }
                }
            }
        }

        // Process any remaining data in the buffer
        let remaining = buffer.trim().to_string();
        if !remaining.is_empty() {
            if let Ok(chunk) = serde_json::from_str::<ChatChunk>(&remaining) {
                if let Some(message) = &chunk.message {
                    if let Some(content) = &message.content {
                        if !content.is_empty() {
                            full_response.push_str(content);
                            let _ = on_chunk(content.clone());
                        }
                    }

                    if let Some(tcs) = &message.tool_calls {
                        for (i, tc) in tcs.iter().enumerate() {
                            let id = format!("ollama_call_{}", tool_calls.len() + i);
                            let arguments = serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_else(|_| "{}".to_string());
                            tool_calls.push(ToolCall::new(
                                id,
                                tc.function.name.clone(),
                                arguments,
                            ));
                        }
                    }
                }
            }
        }

        Ok(build_response(full_response, tool_calls))
    }
}

fn build_response(text: String, tool_calls: Vec<ToolCall>) -> LlmResponse {
    if tool_calls.is_empty() {
        LlmResponse::text(text)
    } else {
        LlmResponse::with_tool_calls(text, tool_calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_from_user() {
        let msg = Message::new(Role::User, "hello");
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "user");
        assert_eq!(chat_msg.content, "hello");
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
        assert_eq!(chat_msg.content, "file contents");
        assert_eq!(chat_msg.tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn chat_message_from_assistant_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "read_file", r#"{"path": "/tmp/a"}"#)];
        let msg = Message::assistant_with_tool_calls(calls);
        let chat_msg = ChatMessage::from(&msg);
        assert_eq!(chat_msg.role, "assistant");
        let tc = chat_msg.tool_calls.unwrap();
        assert_eq!(tc.len(), 1);
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
        let client = OllamaClient::new("http://localhost:11434", "llama3.2")
            .with_model("mistral")
            .with_base_url("http://localhost:9999");
        assert_eq!(client.model, "mistral");
        assert_eq!(client.base_url, "http://localhost:9999");
    }

    #[test]
    fn parse_ndjson_chunk_with_content() {
        let data = r#"{"message":{"role":"assistant","content":"Hello"},"done":false}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert!(!chunk.done);
        let msg = chunk.message.unwrap();
        assert_eq!(msg.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn parse_ndjson_done_chunk() {
        let data = r#"{"message":{"role":"assistant","content":""},"done":true}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert!(chunk.done);
    }

    #[test]
    fn parse_ndjson_chunk_with_tool_calls() {
        let data = r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"/tmp/a"}}}]},"done":false}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        let msg = chunk.message.unwrap();
        let tcs = msg.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "read_file");
        assert_eq!(
            tcs[0].function.arguments,
            serde_json::json!({"path": "/tmp/a"})
        );
    }

    #[test]
    fn request_without_tools_omits_field() {
        let req = ChatRequest {
            model: "llama3.2".into(),
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
            model: "llama3.2".into(),
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
    fn build_response_text_only() {
        let resp = build_response("hello".into(), vec![]);
        assert_eq!(resp.text, "hello");
        assert!(!resp.has_tool_calls());
    }

    #[test]
    fn build_response_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "test", "{}")];
        let resp = build_response("".into(), calls);
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls.len(), 1);
    }

    #[test]
    fn tool_call_arguments_serialized_as_json_string() {
        // Ollama returns arguments as a JSON object, but our ToolCall stores them as a string
        let args = serde_json::json!({"path": "/tmp/a"});
        let serialized = serde_json::to_string(&args).unwrap();
        assert_eq!(serialized, r#"{"path":"/tmp/a"}"#);
    }
}
