use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient};
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
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
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
    /// Optionally reads `OPENAI_MODEL` (defaults to gpt-4o)
    /// and `OPENAI_BASE_URL` (defaults to https://api.openai.com/v1).
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
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

impl From<&Message> for ChatMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        };
        ChatMessage {
            role: role.to_string(),
            content: msg.content.clone(),
        }
    }
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Delta,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
}

impl LlmClient for OpenAiClient {
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        mut on_chunk: ChunkCallback,
    ) -> Result<String, CoreError> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(ChatMessage::from).collect(),
            stream: true,
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
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        });
        let stream_reader = tokio_util::io::StreamReader::new(mapped_stream);
        let mut lines = tokio::io::BufReader::new(stream_reader).lines();

        let mut full_response = String::new();

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
                        if let Some(content) =
                            chunk.choices.first().and_then(|c| c.delta.content.as_ref())
                        {
                            full_response.push_str(content);
                            if !on_chunk(content.clone()) {
                                tracing::debug!("streaming aborted by callback");
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("failed to parse SSE chunk: {e}, data: {data}");
                    }
                }
            }
        }

        Ok(full_response)
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
        assert_eq!(chat_msg.content, "hello");
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
    fn client_builder() {
        let client = OpenAiClient::new("test-key".into())
            .with_model("gpt-3.5-turbo")
            .with_base_url("http://localhost:8080");
        assert_eq!(client.model, "gpt-3.5-turbo");
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn parse_sse_chunk() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_ref().unwrap(), "Hello");
    }

    #[test]
    fn parse_sse_chunk_no_content() {
        let data = r#"{"choices":[{"delta":{}}]}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn from_env_missing_key() {
        // Ensure the env var is not set for this test
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = OpenAiClient::from_env();
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }
}
