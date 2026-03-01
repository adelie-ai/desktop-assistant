use aws_config::{BehaviorVersion, Region};
use aws_credential_types::Credentials;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, Message as BedrockMessage, SystemContentBlock, Tool,
    ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification,
    ToolUseBlock,
};
use aws_smithy_types::{Document, Number};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse};
use std::collections::BTreeMap;
use tokio::sync::OnceCell;

/// Amazon Bedrock client using the Converse API.
pub struct BedrockClient {
    model: String,
    base_url: String,
    api_key: String,
    client: OnceCell<Client>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
}

impl BedrockClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("anthropic.claude-3-5-sonnet-20241022-v2:0")
    }

    pub fn get_default_base_url() -> Option<&'static str> {
        Some("us-east-1")
    }

    pub fn new(api_key: String) -> Self {
        Self {
            model: Self::get_default_model().unwrap_or_default().to_string(),
            base_url: Self::get_default_base_url().unwrap_or_default().to_string(),
            api_key,
            client: OnceCell::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.client = OnceCell::new();
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

    async fn client(&self) -> Result<&Client, CoreError> {
        self.client
            .get_or_try_init(|| async {
                let mut loader = aws_config::defaults(BehaviorVersion::latest());

                if let Some(region) = region_from_base_url(&self.base_url) {
                    loader = loader.region(Region::new(region));
                }

                if let Some(credentials) = static_credentials_from_api_key(&self.api_key) {
                    loader = loader.credentials_provider(credentials);
                } else if !self.api_key.trim().is_empty() {
                    tracing::debug!(
                        "llm.bedrock.api_key is set but not parseable as static credentials; falling back to AWS credential chain"
                    );
                }

                let shared_config = loader.load().await;
                Ok(Client::new(&shared_config))
            })
            .await
    }

    /// Return the model ID as the stable version identifier.
    ///
    /// Bedrock model IDs already include version info (e.g.
    /// `amazon.titan-embed-text-v2:0`), so no server call is needed.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        Ok(self.model.clone())
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let client = self.client().await?;

        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let payload = serde_json::json!({
                "inputText": text,
            });

            let response = client
                .invoke_model()
                .model_id(self.model.clone())
                .content_type("application/json")
                .accept("application/json")
                .body(payload.to_string().into_bytes().into())
                .send()
                .await
                .map_err(|e| CoreError::Llm(format!("Bedrock embeddings request failed: {e}")))?;

            let body = response.body.into_inner();
            let parsed: BedrockEmbeddingResponse = serde_json::from_slice(&body).map_err(|e| {
                CoreError::Llm(format!("failed to parse Bedrock embedding response: {e}"))
            })?;

            vectors.push(parsed.embedding);
        }

        Ok(vectors)
    }
}

#[derive(serde::Deserialize)]
struct BedrockEmbeddingResponse {
    #[serde(default)]
    embedding: Vec<f32>,
}

fn region_from_base_url(base_url: &str) -> Option<String> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return None;
    }

    if !trimmed.contains("http://") && !trimmed.contains("https://") {
        return Some(trimmed.to_string());
    }

    let without_scheme = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);

    let host = without_scheme.split('/').next().unwrap_or_default();
    let segments: Vec<&str> = host.split('.').collect();
    if segments.len() >= 4
        && segments.first().copied() == Some("bedrock-runtime")
        && segments.get(2).copied() == Some("amazonaws")
    {
        return segments.get(1).map(|s| s.to_string());
    }

    None
}

fn static_credentials_from_api_key(api_key: &str) -> Option<Credentials> {
    let trimmed = api_key.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut parts = trimmed.splitn(3, ':');
    let access_key_id = parts.next()?.trim();
    let secret_access_key = parts.next()?.trim();
    let session_token = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);

    if access_key_id.is_empty() || secret_access_key.is_empty() {
        return None;
    }

    Some(Credentials::new(
        access_key_id.to_string(),
        secret_access_key.to_string(),
        session_token,
        None,
        "desktop-assistant-bedrock-static",
    ))
}

fn convert_messages(
    messages: &[Message],
) -> Result<(Vec<SystemContentBlock>, Vec<BedrockMessage>), CoreError> {
    let mut system = Vec::new();
    let mut api_messages = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                system.push(SystemContentBlock::Text(msg.content.clone()));
            }
            Role::User => {
                api_messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(msg.content.clone()))
                        .build()
                        .map_err(|e| {
                            CoreError::Llm(format!(
                                "failed to build Bedrock user message payload: {e}"
                            ))
                        })?,
                );
            }
            Role::Assistant => {
                let mut builder = BedrockMessage::builder().role(ConversationRole::Assistant);

                if !msg.content.is_empty() {
                    builder = builder.content(ContentBlock::Text(msg.content.clone()));
                }

                for tc in &msg.tool_calls {
                    let input_json = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                        .unwrap_or(serde_json::json!({}));
                    let doc = json_to_document(input_json);
                    builder = builder.content(ContentBlock::ToolUse(
                        ToolUseBlock::builder()
                            .tool_use_id(tc.id.clone())
                            .name(tc.name.clone())
                            .input(doc)
                            .build()
                            .map_err(|e| {
                                CoreError::Llm(format!(
                                    "failed to build Bedrock assistant tool-use payload: {e}"
                                ))
                            })?,
                    ));
                }

                api_messages.push(builder.build().map_err(|e| {
                    CoreError::Llm(format!(
                        "failed to build Bedrock assistant message payload: {e}"
                    ))
                })?);
            }
            Role::Tool => {
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                api_messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::ToolResult(
                            ToolResultBlock::builder()
                                .tool_use_id(tool_use_id)
                                .content(ToolResultContentBlock::Text(msg.content.clone()))
                                .build()
                                .map_err(|e| {
                                    CoreError::Llm(format!(
                                        "failed to build Bedrock tool-result payload: {e}"
                                    ))
                                })?,
                        ))
                        .build()
                        .map_err(|e| {
                            CoreError::Llm(format!(
                                "failed to build Bedrock tool message payload: {e}"
                            ))
                        })?,
                );
            }
        }
    }

    Ok((system, api_messages))
}

fn convert_tools(tools: &[ToolDefinition]) -> Result<Option<ToolConfiguration>, CoreError> {
    if tools.is_empty() {
        return Ok(None);
    }

    let mut cfg_builder = ToolConfiguration::builder();
    for tool in tools {
        let input_doc = json_to_document(tool.parameters.clone());
        let spec = ToolSpecification::builder()
            .name(tool.name.clone())
            .description(tool.description.clone())
            .input_schema(ToolInputSchema::Json(input_doc))
            .build()
            .map_err(|e| CoreError::Llm(format!("failed to build Bedrock tool spec: {e}")))?;
        cfg_builder = cfg_builder.tools(Tool::ToolSpec(spec));
    }

    let cfg = cfg_builder
        .build()
        .map_err(|e| CoreError::Llm(format!("failed to build Bedrock tool config: {e}")))?;

    Ok(Some(cfg))
}

#[derive(Default)]
struct ToolCallAccumulator {
    entries: BTreeMap<i32, ToolCallEntry>,
}

#[derive(Default)]
struct ToolCallEntry {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn start_tool_use(&mut self, index: i32, id: impl Into<String>, name: impl Into<String>) {
        let entry = self.entries.entry(index).or_default();
        entry.id = id.into();
        entry.name = name.into();
    }

    fn append_arguments(&mut self, index: i32, chunk: &str) {
        self.entries
            .entry(index)
            .or_default()
            .arguments
            .push_str(chunk);
    }

    fn into_tool_calls(self) -> Vec<ToolCall> {
        self.entries
            .into_values()
            .filter(|entry| !entry.id.is_empty() || !entry.name.is_empty())
            .map(|entry| ToolCall::new(entry.id, entry.name, entry.arguments))
            .collect()
    }
}

fn apply_stream_event(
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
    text: &mut String,
    tool_acc: &mut ToolCallAccumulator,
    on_chunk: &mut ChunkCallback,
) -> bool {
    match event {
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(start) => {
            if let Some(content_start) = start.start()
                && let aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tool_use) =
                    content_start
            {
                tool_acc.start_tool_use(
                    start.content_block_index(),
                    tool_use.tool_use_id(),
                    tool_use.name(),
                );
            }
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(delta) => {
            if let Some(content_delta) = delta.delta() {
                match content_delta {
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(chunk) => {
                        text.push_str(chunk);
                        if !on_chunk(chunk.clone()) {
                            tracing::debug!("Bedrock stream aborted by callback");
                            return false;
                        }
                    }
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tool_delta) => {
                        tool_acc.append_arguments(delta.content_block_index(), tool_delta.input());
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }

    true
}

impl LlmClient for BedrockClient {
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
        let client = self.client().await?;
        let (system, api_messages) = convert_messages(&messages)?;
        let tool_config = convert_tools(tools)?;

        let msg_count = api_messages.len();
        let tool_count = tools.len();
        let system_chars: usize = system.iter().map(|b| format!("{b:?}").len()).sum();
        let msg_chars: usize = api_messages.iter().map(|m| format!("{m:?}").len()).sum();
        tracing::info!(
            msg_chars,
            msg_count,
            tool_count,
            system_chars,
            model = %self.model,
            "LLM request payload"
        );

        let mut request = client
            .converse_stream()
            .model_id(self.model.clone())
            .set_messages(Some(api_messages));

        if self.temperature.is_some() || self.top_p.is_some() || self.max_tokens.is_some() {
            let mut inference_cfg =
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();
            if let Some(t) = self.temperature {
                inference_cfg = inference_cfg.temperature(t as f32);
            }
            if let Some(p) = self.top_p {
                inference_cfg = inference_cfg.top_p(p as f32);
            }
            if let Some(m) = self.max_tokens {
                inference_cfg = inference_cfg.max_tokens(m as i32);
            }
            request = request.inference_config(inference_cfg.build());
        }

        if !system.is_empty() {
            request = request.set_system(Some(system));
        }
        if let Some(cfg) = tool_config {
            request = request.tool_config(cfg);
        }

        let response = request
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("Bedrock converse_stream request failed: {e}")))?;

        let mut stream = response.stream;

        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();

        while let Some(event) = stream
            .recv()
            .await
            .map_err(|e| CoreError::Llm(format!("Bedrock stream receive failed: {e}")))?
        {
            if !apply_stream_event(event, &mut text, &mut tool_acc, &mut on_chunk) {
                break;
            }
        }

        let tool_calls = tool_acc.into_tool_calls();

        if tool_calls.is_empty() {
            Ok(LlmResponse::text(text))
        } else {
            Ok(LlmResponse::with_tool_calls(text, tool_calls))
        }
    }
}

fn json_to_document(value: serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(v) => Document::Bool(v),
        serde_json::Value::String(v) => Document::String(v),
        serde_json::Value::Number(n) => {
            if let Some(v) = n.as_u64() {
                Document::Number(Number::PosInt(v))
            } else if let Some(v) = n.as_i64() {
                Document::Number(Number::NegInt(v))
            } else {
                Document::Number(Number::Float(n.as_f64().unwrap_or_default()))
            }
        }
        serde_json::Value::Array(values) => {
            Document::Array(values.into_iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.into_iter()
                .map(|(k, v)| (k, json_to_document(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::{
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
        ConverseStreamOutput, ToolUseBlockDelta, ToolUseBlockStart,
    };
    use std::sync::{Arc, Mutex};

    #[test]
    fn region_parsing_supports_raw_region() {
        assert_eq!(
            region_from_base_url("us-west-2").as_deref(),
            Some("us-west-2")
        );
    }

    #[test]
    fn region_parsing_supports_bedrock_endpoint() {
        assert_eq!(
            region_from_base_url("https://bedrock-runtime.us-east-1.amazonaws.com").as_deref(),
            Some("us-east-1")
        );
    }

    #[test]
    fn region_parsing_rejects_unknown_endpoint() {
        assert!(region_from_base_url("https://example.com").is_none());
    }

    #[test]
    fn static_credentials_supports_colon_format() {
        let creds = static_credentials_from_api_key("AKIA123:secret456").expect("credentials");
        assert_eq!(creds.access_key_id(), "AKIA123");
        assert_eq!(creds.secret_access_key(), "secret456");
        assert!(creds.session_token().is_none());
    }

    #[test]
    fn static_credentials_supports_session_token() {
        let creds =
            static_credentials_from_api_key("AKIA123:secret456:token789").expect("credentials");
        assert_eq!(creds.access_key_id(), "AKIA123");
        assert_eq!(creds.secret_access_key(), "secret456");
        assert_eq!(creds.session_token(), Some("token789"));
    }

    #[test]
    fn tool_call_accumulator_builds_single_call_from_deltas() {
        let mut acc = ToolCallAccumulator::default();

        acc.start_tool_use(0, "call_1", "read_file");
        acc.append_arguments(0, "{\"path\":\"/tmp");
        acc.append_arguments(0, "/a\"}");

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, "{\"path\":\"/tmp/a\"}");
    }

    #[test]
    fn tool_call_accumulator_orders_calls_by_block_index() {
        let mut acc = ToolCallAccumulator::default();

        acc.start_tool_use(2, "call_2", "tool_b");
        acc.append_arguments(2, "{}");

        acc.start_tool_use(1, "call_1", "tool_a");
        acc.append_arguments(1, "{}");

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "tool_a");
        assert_eq!(calls[1].id, "call_2");
        assert_eq!(calls[1].name, "tool_b");
    }

    #[test]
    fn stream_event_processing_handles_mixed_text_and_tool_calls() {
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let chunks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = Arc::clone(&chunks);

        let mut on_chunk: ChunkCallback = Box::new(move |chunk| {
            chunks_clone.lock().expect("lock").push(chunk);
            true
        });

        let tool_start = ContentBlockStartEvent::builder()
            .content_block_index(0)
            .start(ContentBlockStart::ToolUse(
                ToolUseBlockStart::builder()
                    .tool_use_id("call_1")
                    .name("read_file")
                    .build()
                    .expect("tool start"),
            ))
            .build()
            .expect("start event");

        let text_delta = ContentBlockDeltaEvent::builder()
            .content_block_index(1)
            .delta(ContentBlockDelta::Text("Hello".to_string()))
            .build()
            .expect("text delta");

        let tool_delta_1 = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::ToolUse(
                ToolUseBlockDelta::builder()
                    .input("{\"path\":\"/tmp")
                    .build()
                    .expect("tool delta 1"),
            ))
            .build()
            .expect("tool delta event 1");

        let tool_delta_2 = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::ToolUse(
                ToolUseBlockDelta::builder()
                    .input("/a\"}")
                    .build()
                    .expect("tool delta 2"),
            ))
            .build()
            .expect("tool delta event 2");

        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockStart(tool_start),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
        ));
        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(text_delta),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
        ));
        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(tool_delta_1),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
        ));
        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(tool_delta_2),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
        ));

        assert_eq!(text, "Hello");
        assert_eq!(*chunks.lock().expect("lock"), vec!["Hello"]);

        let calls = tool_acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, "{\"path\":\"/tmp/a\"}");
    }

    #[test]
    fn stream_event_processing_stops_on_callback_abort() {
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut seen = 0usize;

        let mut on_chunk: ChunkCallback = Box::new(move |_chunk| {
            seen += 1;
            seen < 2
        });

        let first = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::Text("A".to_string()))
            .build()
            .expect("first delta");
        let second = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::Text("B".to_string()))
            .build()
            .expect("second delta");

        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(first),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
        ));
        assert!(!apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(second),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
        ));
        assert_eq!(text, "AB");
    }
}
