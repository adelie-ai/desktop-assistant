use std::collections::HashMap;
use std::sync::Mutex;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ReasoningConfig,
    TokenUsage, current_model_override,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;

/// Ollama LLM client that streams completions via the native `/api/chat` endpoint.
///
/// Uses NDJSON streaming (one JSON object per line) and Ollama's native tool
/// calling format. No authentication is required.
///
/// Context windows are not curated — Ollama hosts arbitrary GGUF models, so
/// we read the value from the per-model `POST /api/show` response. Because
/// `LlmClient::max_context_tokens` is synchronous and the source is an HTTP
/// call, the connector caches results per-model id in
/// [`OllamaClient::context_length_cache`]. Callers should invoke
/// [`OllamaClient::warm_context_length`] (fire-and-forget) shortly after
/// construction to populate the cache for `self.model`; until then,
/// `max_context_tokens()` returns `None` and the daemon's universal
/// fallback applies. The cache is keyed by model id so per-turn model
/// overrides (issue #34) can be warmed independently — but a cold lookup
/// for an overridden model still returns `None` until that model has been
/// warmed.
pub struct OllamaClient {
    client: Client,
    model: String,
    base_url: String,
    model_ready: OnceCell<()>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    /// Per-model cache of `/api/show`-derived context lengths. `None`
    /// values are cached too (when `/api/show` declines to populate the
    /// field) so we don't keep retrying. Populated by
    /// [`Self::warm_context_length`] and [`Self::context_length_for`].
    context_length_cache: Mutex<HashMap<String, Option<u64>>>,
}

impl OllamaClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("llama3.2")
    }

    pub fn get_default_base_url() -> Option<&'static str> {
        Some("http://localhost:11434")
    }

    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into(),
            model_ready: OnceCell::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            context_length_cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.model_ready = OnceCell::new();
        self.context_length_cache = Mutex::new(HashMap::new());
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self.model_ready = OnceCell::new();
        self.context_length_cache = Mutex::new(HashMap::new());
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

    /// Return the model name stamped with the server-side digest.
    ///
    /// Calls `GET {base_url}/api/tags` and finds the matching model's digest,
    /// returning `"{model}@{digest}"`.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("model tags HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "Ollama tags API error (HTTP {status}): {text}"
            )));
        }

        let tags: OllamaTagsResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse tags response: {e}")))?;

        let digest = tags
            .models
            .iter()
            .find(|m| model_matches(&self.model, m))
            .and_then(|m| m.digest.as_deref())
            .ok_or_else(|| {
                CoreError::Llm(format!(
                    "model '{}' not found in Ollama tags response",
                    self.model
                ))
            })?;

        Ok(format!("{}@{}", self.model, digest))
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
            "truncate": true,
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
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
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
    digest: Option<String>,
}

/// Partial decoding of `POST /api/show` used to pluck the context window.
///
/// Ollama reports context length under `model_info["<arch>.context_length"]`
/// where `<arch>` varies by model family (`llama`, `qwen2`, `gemma3`, etc).
/// We scan every `*.context_length` key and take the first `u64` we find.
#[derive(Deserialize, Default)]
struct OllamaShowResponse {
    #[serde(default)]
    model_info: std::collections::BTreeMap<String, serde_json::Value>,
}

impl OllamaShowResponse {
    fn context_length(&self) -> Option<u64> {
        self.model_info
            .iter()
            .find(|(k, _)| k.ends_with(".context_length") || k.as_str() == "context_length")
            .and_then(|(_, v)| v.as_u64())
    }
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
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
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

impl OllamaClient {
    /// List models installed on the connected Ollama server.
    ///
    /// Calls `GET /api/tags` to enumerate installed models and, for each,
    /// `POST /api/show` to extract the context window declared by the
    /// model's GGUF metadata. `/api/show` is best-effort: failures are
    /// logged and the model is still returned without a `context_limit`.
    async fn list_models_impl(&self) -> Result<Vec<ModelInfo>, CoreError> {
        let base_url = self.base_url.trim_end_matches('/');
        let tags_url = format!("{base_url}/api/tags");

        let response = self
            .client
            .get(&tags_url)
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("Ollama /api/tags request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(CoreError::Llm(format!(
                "Ollama /api/tags error (HTTP {status}): {body}"
            )));
        }

        let tags: OllamaTagsResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse Ollama /api/tags: {e}")))?;

        let mut models = Vec::with_capacity(tags.models.len());
        for tag in tags.models {
            let display = tag.model.clone().unwrap_or_else(|| tag.name.clone());
            let context_limit = self.fetch_context_limit(base_url, &tag.name).await;

            // Ollama is local inference for open-source chat models.
            // Tools support is widespread on modern models, but the API
            // doesn't expose a reliable capability flag — default to true
            // and let callers fall back on 400 responses. Embedding-only
            // models are recognised by an `embed` token in the id (matches
            // the `*-embed-*` and `*-embedding` naming used by every Ollama
            // embedding model on https://ollama.com/library) so the picker
            // can filter them correctly for the embedding purpose.
            let lower_id = tag.name.to_ascii_lowercase();
            let is_embedding = lower_id.contains("embed");
            let capabilities = ModelCapabilities {
                reasoning: false,
                vision: false,
                tools: !is_embedding,
                embedding: is_embedding,
            };

            models.push(ModelInfo {
                id: tag.name,
                display_name: display,
                context_limit,
                capabilities,
            });
        }

        Ok(models)
    }

    async fn fetch_context_limit(&self, base_url: &str, model: &str) -> Option<u64> {
        let show_url = format!("{base_url}/api/show");
        let body = serde_json::json!({ "model": model });

        let response = match self.client.post(&show_url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(
                    model,
                    "Ollama /api/show request failed: {e}; leaving context_limit unset"
                );
                return None;
            }
        };

        if !response.status().is_success() {
            tracing::debug!(
                model,
                status = %response.status(),
                "Ollama /api/show non-success; leaving context_limit unset"
            );
            return None;
        }

        let show: OllamaShowResponse = match response.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(model, "failed to parse /api/show response: {e}");
                return None;
            }
        };

        show.context_length()
    }

    /// Populate the context-length cache for `self.model` by calling
    /// `/api/show` and parsing the GGUF metadata. Safe to call as
    /// fire-and-forget: failures (server down, model not pulled, malformed
    /// response) are logged at `debug` and the cache stays empty so a
    /// subsequent call retries.
    ///
    /// Returns the same value that [`Self::max_context_tokens`] will report
    /// after this call returns. Idempotent across concurrent invocations:
    /// the underlying `/api/show` call may run more than once if races
    /// occur, but the last writer wins and all readers see a consistent
    /// value once writes settle.
    pub async fn warm_context_length(&self) -> Option<u64> {
        self.warm_context_length_for(&self.model).await
    }

    /// Warm the context-length cache for an arbitrary model id (used when
    /// a per-turn override targets a model other than `self.model`).
    pub async fn warm_context_length_for(&self, model: &str) -> Option<u64> {
        let base_url = self.base_url.trim_end_matches('/');
        let value = self.fetch_context_limit(base_url, model).await;
        if let Ok(mut guard) = self.context_length_cache.lock() {
            guard.insert(model.to_string(), value);
        }
        value
    }

    /// Look up a cached context-length value for `model` without firing a
    /// network request. Returns `None` when the model has not been warmed
    /// yet *or* when `/api/show` previously declined to populate the
    /// field; callers can't distinguish the two cases (the daemon falls
    /// back to its universal default in either case).
    fn cached_context_length(&self, model: &str) -> Option<u64> {
        self.context_length_cache
            .lock()
            .ok()
            .and_then(|guard| guard.get(model).copied().flatten())
    }
}

impl LlmClient for OllamaClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        let model = current_model_override().unwrap_or_else(|| self.model.clone());
        self.cached_context_length(&model)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.list_models_impl().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // Ollama exposes no standardized reasoning/thinking knob across
        // community models; log at debug and otherwise ignore. See #18.
        if !reasoning.is_empty() {
            tracing::debug!(
                model = %self.model,
                ?reasoning,
                "reasoning hint ignored on Ollama connector (no-op)"
            );
        }
        self.ensure_model_available().await?;

        // Per-turn model override (issue #34): when the daemon-side routing
        // layer has set `MODEL_OVERRIDE`, dispatch the user-chosen model
        // instead of the connector's baked-in `self.model`. Note that the
        // pre-flight `ensure_model_available()` above still keys on
        // `self.model` (the connection's default); when overriding to a
        // different local model, Ollama itself surfaces a clean error if
        // it isn't pulled.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        let chat_tools: Vec<ChatTool> = tools.iter().map(ChatTool::from).collect();

        let options =
            if self.temperature.is_some() || self.top_p.is_some() || self.max_tokens.is_some() {
                Some(OllamaOptions {
                    temperature: self.temperature,
                    top_p: self.top_p,
                    num_predict: self.max_tokens,
                })
            } else {
                None
            };

        let request = ChatRequest {
            model,
            messages: messages.iter().map(ChatMessage::from).collect(),
            stream: true,
            tools: chat_tools,
            options,
        };

        let request_json =
            serde_json::to_string(&request).unwrap_or_else(|_| "<serialization error>".into());
        let request_bytes = request_json.len();
        let msg_count = request.messages.len();
        let tool_count = request.tools.len();
        tracing::info!(
            request_bytes,
            msg_count,
            tool_count,
            model = %request.model,
            "LLM request payload"
        );
        tracing::debug!(
            "LLM request body (first 2000 chars): {}",
            &request_json[..request_json.len().min(2000)]
        );

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
        let mut token_usage: Option<TokenUsage> = None;
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
                                    return Ok(build_response(
                                        full_response,
                                        tool_calls,
                                        token_usage,
                                    ));
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
                            if chunk.prompt_eval_count.is_some() || chunk.eval_count.is_some() {
                                token_usage = Some(TokenUsage {
                                    input_tokens: chunk.prompt_eval_count,
                                    output_tokens: chunk.eval_count,
                                    ..Default::default()
                                });
                            }
                            return Ok(build_response(full_response, tool_calls, token_usage));
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
        {
            if let Some(message) = &chunk.message {
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

            if chunk.done && (chunk.prompt_eval_count.is_some() || chunk.eval_count.is_some()) {
                token_usage = Some(TokenUsage {
                    input_tokens: chunk.prompt_eval_count,
                    output_tokens: chunk.eval_count,
                    ..Default::default()
                });
            }
        }

        Ok(build_response(full_response, tool_calls, token_usage))
    }
}

fn build_response(
    text: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<TokenUsage>,
) -> LlmResponse {
    let response = if tool_calls.is_empty() {
        LlmResponse::text(text)
    } else {
        LlmResponse::with_tool_calls(text, tool_calls)
    };
    match usage {
        Some(u) => response.with_usage(u),
        None => response,
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
            options: None,
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
            options: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"function\""));
        assert!(json.contains("\"test\""));
    }

    #[test]
    fn build_response_text_only() {
        let resp = build_response("hello".into(), vec![], None);
        assert_eq!(resp.text, "hello");
        assert!(!resp.has_tool_calls());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn build_response_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "test", "{}")];
        let resp = build_response("".into(), calls, None);
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls.len(), 1);
    }

    #[test]
    fn parse_done_chunk_with_eval_counts() {
        let data = r#"{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":42,"eval_count":17}"#;
        let chunk: ChatChunk = serde_json::from_str(data).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.prompt_eval_count, Some(42));
        assert_eq!(chunk.eval_count, Some(17));
    }

    #[test]
    fn tool_call_arguments_serialized_as_json_string() {
        // Ollama returns arguments as a JSON object, but our ToolCall stores them as a string
        let args = serde_json::json!({"path": "/tmp/a"});
        let serialized = serde_json::to_string(&args).unwrap();
        assert_eq!(serialized, r#"{"path":"/tmp/a"}"#);
    }

    #[tokio::test]
    async fn stream_completion_uses_self_model_when_override_unset() {
        // Issue #34 negative case: with no `MODEL_OVERRIDE` task-local set,
        // dispatch must use the connector's baked-in `self.model`.
        let server = MockServer::start();

        let _tags = server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"llama3.2:latest"}]}"#);
        });
        let chat = server.mock(|when, then| {
            when.method(POST)
                .path("/api/chat")
                .body_includes(r#""model":"llama3.2""#);
            then.status(200)
                .header("content-type", "application/x-ndjson")
                .body("{\"message\":{\"content\":\"ok\"},\"done\":true}\n");
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");
        client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("dispatch must succeed");
        chat.assert_calls(1);
    }

    #[tokio::test]
    async fn stream_completion_uses_model_override_when_set() {
        // Issue #34 happy path: with `MODEL_OVERRIDE` task-local set, the
        // chat request body carries the override model id rather than
        // `self.model`.
        use desktop_assistant_core::ports::llm::with_model_override;

        let server = MockServer::start();

        let _tags = server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"llama3.2:latest"}]}"#);
        });
        let chat = server.mock(|when, then| {
            when.method(POST)
                .path("/api/chat")
                .body_includes(r#""model":"qwen3""#);
            then.status(200)
                .header("content-type", "application/x-ndjson")
                .body("{\"message\":{\"content\":\"ok\"},\"done\":true}\n");
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");
        with_model_override("qwen3".into(), async {
            client
                .stream_completion(
                    vec![Message::new(Role::User, "hi")],
                    &[],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await
                .expect("dispatch must succeed with override");
        })
        .await;
        chat.assert_calls(1);
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
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response_first.text, "Hello");

        let response_second = client
            .stream_completion(
                vec![Message::new(Role::User, "again")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response_second.text, "Hello");

        tags_mock.assert_calls(1);
        pull_mock.assert_calls(1);
        chat_mock.assert_calls(2);
    }

    #[tokio::test]
    async fn model_identifier_returns_name_at_digest() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"nomic-embed-text:latest","model":"nomic-embed-text:latest","digest":"sha256:abcdef1234567890"}]}"#);
        });

        let client = OllamaClient::new(server.url(""), "nomic-embed-text");
        let id = client.model_identifier().await.unwrap();
        assert_eq!(id, "nomic-embed-text@sha256:abcdef1234567890");
    }

    #[tokio::test]
    async fn list_models_returns_tags_enriched_with_context_limit() {
        let server = MockServer::start();

        let tags_mock = server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"models":[
                        {"name":"llama3.2:latest","model":"llama3.2:latest","digest":"sha256:aaa"},
                        {"name":"nomic-embed-text:latest","model":"nomic-embed-text:latest","digest":"sha256:bbb"}
                    ]}"#,
                );
        });

        let show_llama = server.mock(|when, then| {
            when.method(POST)
                .path("/api/show")
                .body_includes("llama3.2:latest");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"llama.context_length":131072}}"#);
        });

        let show_embed = server.mock(|when, then| {
            when.method(POST)
                .path("/api/show")
                .body_includes("nomic-embed-text:latest");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"nomic-bert.context_length":8192}}"#);
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");
        let models = client.list_models().await.unwrap();

        assert_eq!(models.len(), 2);
        let llama = models.iter().find(|m| m.id == "llama3.2:latest").unwrap();
        assert_eq!(llama.context_limit, Some(131_072));
        assert!(llama.capabilities.tools);
        let embed = models
            .iter()
            .find(|m| m.id == "nomic-embed-text:latest")
            .unwrap();
        assert_eq!(embed.context_limit, Some(8_192));

        tags_mock.assert_calls(1);
        show_llama.assert_calls(1);
        show_embed.assert_calls(1);
    }

    #[tokio::test]
    async fn list_models_skips_context_limit_when_show_fails() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"mystery:latest"}]}"#);
        });

        // /api/show returns 500 → context_limit should end up None.
        server.mock(|when, then| {
            when.method(POST).path("/api/show");
            then.status(500).body("boom");
        });

        let client = OllamaClient::new(server.url(""), "mystery");
        let models = client.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mystery:latest");
        assert_eq!(models[0].context_limit, None);
    }

    #[tokio::test]
    async fn list_models_returns_empty_when_no_models_installed() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method(GET).path("/api/tags");
            then.status(200).body(r#"{"models":[]}"#);
        });

        let client = OllamaClient::new(server.url(""), "whatever");
        let models = client.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn ollama_show_extracts_context_length_for_any_arch() {
        let show: OllamaShowResponse = serde_json::from_str(
            r#"{"model_info":{"qwen2.context_length":32768,"qwen2.vocab_size":152064}}"#,
        )
        .unwrap();
        assert_eq!(show.context_length(), Some(32_768));
    }

    #[test]
    fn ollama_show_returns_none_when_no_context_length() {
        let show: OllamaShowResponse =
            serde_json::from_str(r#"{"model_info":{"general.architecture":"llama"}}"#).unwrap();
        assert_eq!(show.context_length(), None);
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

        tags_mock.assert_calls(1);
        pull_mock.assert_calls(1);
        embed_mock.assert_calls(2);
    }

    // --- max_context_tokens / context-length cache tests ---

    #[test]
    fn max_context_tokens_is_none_before_warmup() {
        let client = OllamaClient::new("http://localhost:11434", "llama3.2");
        assert_eq!(client.max_context_tokens(), None);
    }

    #[tokio::test]
    async fn warm_context_length_populates_cache() {
        let server = MockServer::start();
        let show = server.mock(|when, then| {
            when.method(POST).path("/api/show").body_includes("llama3.2");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"llama.context_length":131072}}"#);
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");
        let warmed = client.warm_context_length().await;
        assert_eq!(warmed, Some(131_072));
        assert_eq!(client.max_context_tokens(), Some(131_072));
        show.assert_calls(1);
    }

    #[tokio::test]
    async fn warm_context_length_caches_failures_as_none() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/show");
            then.status(500).body("boom");
        });

        let client = OllamaClient::new(server.url(""), "mystery");
        let warmed = client.warm_context_length().await;
        assert_eq!(warmed, None);
        assert_eq!(client.max_context_tokens(), None);
    }

    #[tokio::test]
    async fn max_context_tokens_consults_model_override() {
        use desktop_assistant_core::ports::llm::with_model_override;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/show").body_includes("llama3.2");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"llama.context_length":131072}}"#);
        });
        server.mock(|when, then| {
            when.method(POST)
                .path("/api/show")
                .body_includes("qwen2:latest");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"qwen2.context_length":32768}}"#);
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");

        // Warm both models — `warm_context_length` defaults to `self.model`,
        // and `warm_context_length_for` covers the override target.
        let _ = client.warm_context_length().await;
        let _ = client.warm_context_length_for("qwen2:latest").await;

        assert_eq!(client.max_context_tokens(), Some(131_072));
        let observed = with_model_override("qwen2:latest".into(), async {
            client.max_context_tokens()
        })
        .await;
        assert_eq!(observed, Some(32_768));
    }

    #[tokio::test]
    async fn max_context_tokens_for_uncached_override_is_none() {
        use desktop_assistant_core::ports::llm::with_model_override;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/api/show").body_includes("llama3.2");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"llama.context_length":131072}}"#);
        });

        let client = OllamaClient::new(server.url(""), "llama3.2");
        let _ = client.warm_context_length().await;

        // Override targets a model that hasn't been warmed: returns None
        // (cache miss is the safe answer).
        let observed = with_model_override("never-warmed".into(), async {
            client.max_context_tokens()
        })
        .await;
        assert_eq!(observed, None);
    }
}
