use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

/// Ollama LLM client that streams completions via the native `/api/chat` endpoint.
///
/// Uses NDJSON streaming (one JSON object per line) and Ollama's native tool
/// calling format. No authentication is required.
pub struct OllamaClient {
    client: Client,
    model: String,
    base_url: String,
    model_ready: OnceCell<()>,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into(),
            model_ready: OnceCell::new(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.model_ready = OnceCell::new();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self.model_ready = OnceCell::new();
        self
    }

    async fn ensure_model_available(&self) -> Result<(), CoreError> {
        self.model_ready
            .get_or_try_init(|| async { self.ensure_model_available_impl().await })
            .await
            .map(|_| ())
    }

    async fn ensure_model_available_impl(&self) -> Result<(), CoreError> {
        let base_url = self.base_url.trim_end_matches('/');
        let tags_url = format!("{base_url}/api/tags");

        let tags_response = self
            .client
            .get(&tags_url)
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to check Ollama models: {e}")))?;

        if !tags_response.status().is_success() {
            let status = tags_response.status();
            let body = tags_response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "Ollama model list API error (HTTP {status}): {body}"
            )));
        }

        let tags: OllamaTagsResponse = tags_response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse Ollama model list: {e}")))?;

        if tags
            .models
            .iter()
            .any(|installed| model_matches(&self.model, installed))
        {
            return Ok(());
        }

        tracing::info!(model = %self.model, "ollama model missing locally; pulling");

        let pull_url = format!("{base_url}/api/pull");
        let pull_response = self
            .client
            .post(&pull_url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "stream": false,
            }))
            .send()
            .await
            .map_err(|e| {
                CoreError::Llm(format!("failed to pull Ollama model '{}': {e}", self.model))
            })?;

        if !pull_response.status().is_success() {
            let status = pull_response.status();
            let body = pull_response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "Ollama model pull API error for '{}' (HTTP {status}): {body}",
                self.model
            )));
        }

        Ok(())
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// Sends a `POST {base_url}/api/embed` request and returns one vector per input.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        self.ensure_model_available().await?;

        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let response = self
            .client
            .post(&url)
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
                "Ollama embeddings API error (HTTP {status}): {body}"
            )));
        }

        let parsed: OllamaEmbedResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse embedding response: {e}")))?;

        Ok(parsed.embeddings)
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
                        let arguments: serde_json::Value = serde_json::from_str(&tc.arguments)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
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

// --- Embedding response types ---

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaModelTag>,
}

#[derive(Deserialize)]
struct OllamaModelTag {
    name: String,
    model: Option<String>,
}

fn model_matches(configured: &str, installed: &OllamaModelTag) -> bool {
    model_name_matches(configured, &installed.name)
        || installed
            .model
            .as_deref()
            .is_some_and(|model| model_name_matches(configured, model))
}

fn model_name_matches(configured: &str, candidate: &str) -> bool {
    configured == candidate
        || (!configured.contains(':') && candidate == format!("{configured}:latest"))
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
        self.ensure_model_available().await?;

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
            buffer.push_str(&String::from_utf8_lossy(&bytes));

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
                            if let Some(content) = &message.content
                                && !content.is_empty()
                            {
                                full_response.push_str(content);
                                if !on_chunk(content.clone()) {
                                    tracing::debug!("streaming aborted by callback");
                                    return Ok(build_response(full_response, tool_calls));
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
        if !remaining.is_empty()
            && let Ok(chunk) = serde_json::from_str::<ChatChunk>(&remaining)
            && let Some(message) = &chunk.message
        {
            if let Some(content) = &message.content
                && !content.is_empty()
            {
                full_response.push_str(content);
                let _ = on_chunk(content.clone());
            }

            if let Some(tcs) = &message.tool_calls {
                for (i, tc) in tcs.iter().enumerate() {
                    let id = format!("ollama_call_{}", tool_calls.len() + i);
                    let arguments = serde_json::to_string(&tc.function.arguments)
                        .unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(ToolCall::new(id, tc.function.name.clone(), arguments));
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
    use httpmock::Method::{GET, POST};
    use httpmock::MockServer;

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
    fn model_name_matches_exact() {
        assert!(model_name_matches("llama3.2", "llama3.2"));
    }

    #[test]
    fn model_name_matches_latest_tag_for_untagged_model() {
        assert!(model_name_matches("llama3.2", "llama3.2:latest"));
    }

    #[test]
    fn model_name_does_not_match_different_tag_when_configured_tagged() {
        assert!(!model_name_matches("llama3.2:8b", "llama3.2:latest"));
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

    #[tokio::test]
    async fn stream_completion_pulls_missing_model_once_before_chat() {
        let server = MockServer::start();

        let tags_mock = server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"other-model:latest"}]}"#);
        });

        let pull_mock = server.mock(|when, then| {
            when.method(POST).path("/api/pull");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"status":"success"}"#);
        });

        let chat_mock = server.mock(|when, then| {
            when.method(POST).path("/api/chat");
            then.status(200)
                .header("content-type", "application/x-ndjson")
                .body(
                    r#"{"message":{"content":"Hello"},"done":true}
"#,
                );
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");

        let response_first = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response_first.text, "Hello");

        let response_second = client
            .stream_completion(
                vec![Message::new(Role::User, "again")],
                &[],
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response_second.text, "Hello");

        tags_mock.assert_hits(1);
        pull_mock.assert_hits(1);
        chat_mock.assert_hits(2);
    }

    #[tokio::test]
    async fn embed_pulls_missing_model_once_before_embed() {
        let server = MockServer::start();

        let tags_mock = server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"another-model:latest"}]}"#);
        });

        let pull_mock = server.mock(|when, then| {
            when.method(POST).path("/api/pull");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"status":"success"}"#);
        });

        let embed_mock = server.mock(|when, then| {
            when.method(POST).path("/api/embed");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"embeddings":[[0.1,0.2],[0.3,0.4]]}"#);
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");

        let first = client
            .embed(vec!["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(first, vec![vec![0.1_f32, 0.2_f32], vec![0.3_f32, 0.4_f32]]);

        let second = client.embed(vec!["c".to_string()]).await.unwrap();
        assert_eq!(second, vec![vec![0.1_f32, 0.2_f32], vec![0.3_f32, 0.4_f32]]);

        tags_mock.assert_hits(1);
        pull_mock.assert_hits(1);
        embed_mock.assert_hits(2);
    }
}
