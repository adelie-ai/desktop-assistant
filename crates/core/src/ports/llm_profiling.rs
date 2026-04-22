use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use crate::CoreError;
use crate::domain::{Message, Role, ToolDefinition, ToolNamespace};
use crate::ports::llm::{ChunkCallback, LlmClient, LlmResponse, ModelInfo, TokenUsage};

/// JSONL profiling entry written for each LLM call.
#[derive(Serialize)]
struct ProfileEntry {
    timestamp: String,
    message_count: usize,
    tool_count: usize,
    tool_names: Vec<String>,
    messages: Vec<ProfileMessage>,
    response_text_len: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_text_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_text: Option<String>,
    response_tool_calls: Vec<ProfileToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<TokenUsage>,
    duration_ms: u128,
}

#[derive(Serialize)]
struct ProfileMessage {
    role: String,
    content_len: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ProfileToolCall {
    id: String,
    name: String,
    arguments_len: usize,
}

/// Decorator that captures full request/response context and writes JSONL.
pub struct ProfilingLlmClient<L> {
    inner: L,
    log_path: PathBuf,
    full_content: bool,
}

impl<L> ProfilingLlmClient<L> {
    pub fn new(inner: L, log_path: PathBuf, full_content: bool) -> Self {
        Self {
            inner,
            log_path,
            full_content,
        }
    }

    fn profile_messages(&self, messages: &[Message]) -> Vec<ProfileMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };
                let content_len = m.content.len();
                let (content_preview, content) = if self.full_content {
                    (None, Some(m.content.clone()))
                } else {
                    let preview = if content_len > 200 {
                        format!("{}...", &m.content[..200])
                    } else {
                        m.content.clone()
                    };
                    (Some(preview), None)
                };
                ProfileMessage {
                    role: role.to_string(),
                    content_len,
                    content_preview,
                    content,
                }
            })
            .collect()
    }

    fn log_result(
        &self,
        result: &Result<LlmResponse, CoreError>,
        message_count: usize,
        tool_count: usize,
        tool_names: Vec<String>,
        messages: Vec<ProfileMessage>,
        duration_ms: u128,
    ) {
        match result {
            Ok(response) => {
                let response_text_len = response.text.len();
                let (response_text_preview, response_text) = if self.full_content {
                    (None, Some(response.text.clone()))
                } else {
                    let preview = if response_text_len > 200 {
                        format!("{}...", &response.text[..200])
                    } else {
                        response.text.clone()
                    };
                    (Some(preview), None)
                };

                let entry = ProfileEntry {
                    timestamp: chrono::Utc::now()
                        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                    message_count,
                    tool_count,
                    tool_names,
                    messages,
                    response_text_len,
                    response_text_preview,
                    response_text,
                    response_tool_calls: response
                        .tool_calls
                        .iter()
                        .map(|tc| ProfileToolCall {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments_len: tc.arguments.len(),
                        })
                        .collect(),
                    usage: response.usage.clone(),
                    duration_ms,
                };
                self.write_entry(&entry);
            }
            Err(_) => {
                let entry = ProfileEntry {
                    timestamp: chrono::Utc::now()
                        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                    message_count,
                    tool_count,
                    tool_names,
                    messages,
                    response_text_len: 0,
                    response_text_preview: None,
                    response_text: None,
                    response_tool_calls: vec![],
                    usage: None,
                    duration_ms,
                };
                self.write_entry(&entry);
            }
        }
    }

    fn write_entry(&self, entry: &ProfileEntry) {
        use std::io::Write;
        match serde_json::to_string(entry) {
            Ok(json) => {
                let result = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.log_path)
                    .and_then(|mut f| writeln!(f, "{json}"));
                if let Err(e) = result {
                    tracing::warn!("failed to write LLM profile entry: {e}");
                }
            }
            Err(e) => {
                tracing::warn!("failed to serialize LLM profile entry: {e}");
            }
        }
    }
}

impl<L: LlmClient> LlmClient for ProfilingLlmClient<L> {
    fn get_default_model(&self) -> Option<&str> {
        self.inner.get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.inner.get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        self.inner.max_context_tokens()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.refresh_models().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
        let tool_count = tools.len();
        let message_count = messages.len();
        let profile_messages = self.profile_messages(&messages);

        let start = Instant::now();
        let result = self
            .inner
            .stream_completion(messages, tools, on_chunk)
            .await;
        let duration_ms = start.elapsed().as_millis();

        self.log_result(
            &result,
            message_count,
            tool_count,
            tool_names,
            profile_messages,
            duration_ms,
        );

        result
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.inner.supports_hosted_tool_search()
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let mut all_names: Vec<String> = core_tools.iter().map(|t| t.name.clone()).collect();
        for ns in namespaces {
            for t in &ns.tools {
                all_names.push(t.name.clone());
            }
        }
        let tool_count = all_names.len();
        let message_count = messages.len();
        let profile_messages = self.profile_messages(&messages);

        let start = Instant::now();
        let result = self
            .inner
            .stream_completion_with_namespaces(messages, core_tools, namespaces, on_chunk)
            .await;
        let duration_ms = start.elapsed().as_millis();

        self.log_result(
            &result,
            message_count,
            tool_count,
            all_names,
            profile_messages,
            duration_ms,
        );

        result
    }
}

/// Wrapper enum that conditionally applies profiling.
pub enum MaybeProfiled<L> {
    Plain(L),
    Profiled(ProfilingLlmClient<L>),
}

impl<L> MaybeProfiled<L> {
    /// Check `LLM_PROFILE_LOG` env var; if set, wrap with profiling.
    pub fn from_env(inner: L) -> Self {
        match std::env::var("LLM_PROFILE_LOG") {
            Ok(path) if !path.is_empty() => {
                let full_content = std::env::var("LLM_PROFILE_FULL")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                tracing::info!("LLM profiling enabled → {path}");
                Self::Profiled(ProfilingLlmClient::new(
                    inner,
                    PathBuf::from(path),
                    full_content,
                ))
            }
            _ => Self::Plain(inner),
        }
    }

    /// Build from config values with env var override.
    ///
    /// Precedence: `LLM_PROFILE_LOG` env var → config `enabled` → off.
    pub fn from_config(
        inner: L,
        enabled: bool,
        log_path: Option<&str>,
        full_content: bool,
    ) -> Self {
        // Env var overrides config entirely (backwards compat).
        if let Ok(env_path) = std::env::var("LLM_PROFILE_LOG") {
            if !env_path.is_empty() {
                let env_full = std::env::var("LLM_PROFILE_FULL")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                tracing::info!("LLM profiling enabled (env) → {env_path}");
                return Self::Profiled(ProfilingLlmClient::new(
                    inner,
                    PathBuf::from(env_path),
                    env_full,
                ));
            }
        }

        if !enabled {
            return Self::Plain(inner);
        }

        let resolve_tilde = |p: &str| -> PathBuf {
            if p.starts_with("~/") {
                if let Ok(home) = std::env::var("HOME") {
                    return PathBuf::from(home).join(&p[2..]);
                }
            }
            PathBuf::from(p)
        };

        let path = log_path.map(|p| resolve_tilde(p)).unwrap_or_else(|| {
            let data_dir = std::env::var("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
                    PathBuf::from(home).join(".local/share")
                });
            data_dir.join("desktop-assistant/llm-profile.jsonl")
        });

        tracing::info!("LLM profiling enabled (config) → {}", path.display());
        Self::Profiled(ProfilingLlmClient::new(inner, path, full_content))
    }
}

impl<L: LlmClient> LlmClient for MaybeProfiled<L> {
    fn get_default_model(&self) -> Option<&str> {
        match self {
            Self::Plain(l) => l.get_default_model(),
            Self::Profiled(l) => l.get_default_model(),
        }
    }

    fn get_default_base_url(&self) -> Option<&str> {
        match self {
            Self::Plain(l) => l.get_default_base_url(),
            Self::Profiled(l) => l.get_default_base_url(),
        }
    }

    fn max_context_tokens(&self) -> Option<u64> {
        match self {
            Self::Plain(l) => l.max_context_tokens(),
            Self::Profiled(l) => l.max_context_tokens(),
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        match self {
            Self::Plain(l) => l.list_models().await,
            Self::Profiled(l) => l.list_models().await,
        }
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        match self {
            Self::Plain(l) => l.refresh_models().await,
            Self::Profiled(l) => l.refresh_models().await,
        }
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match self {
            Self::Plain(l) => l.stream_completion(messages, tools, on_chunk).await,
            Self::Profiled(l) => l.stream_completion(messages, tools, on_chunk).await,
        }
    }

    fn supports_hosted_tool_search(&self) -> bool {
        match self {
            Self::Plain(l) => l.supports_hosted_tool_search(),
            Self::Profiled(l) => l.supports_hosted_tool_search(),
        }
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match self {
            Self::Plain(l) => {
                l.stream_completion_with_namespaces(messages, core_tools, namespaces, on_chunk)
                    .await
            }
            Self::Profiled(l) => {
                l.stream_completion_with_namespaces(messages, core_tools, namespaces, on_chunk)
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    struct MockLlm;

    impl LlmClient for MockLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            Ok(LlmResponse::text("mock response").with_usage(TokenUsage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                ..Default::default()
            }))
        }
    }

    #[tokio::test]
    async fn profiling_client_writes_jsonl() {
        let dir = std::env::temp_dir().join(format!("llm_profile_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("profile.jsonl");

        let client = ProfilingLlmClient::new(MockLlm, log_path.clone(), false);

        let response = client
            .stream_completion(
                vec![
                    Message::new(Role::System, "You are helpful"),
                    Message::new(Role::User, "Hello"),
                ],
                &[ToolDefinition::new(
                    "read_file",
                    "Read a file",
                    serde_json::json!({"type": "object"}),
                )],
                Box::new(|_| true),
            )
            .await
            .unwrap();

        assert_eq!(response.text, "mock response");
        assert!(response.usage.is_some());

        let content = std::fs::read_to_string(&log_path).unwrap();
        let entry: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(entry["message_count"], 2);
        assert_eq!(entry["tool_count"], 1);
        assert_eq!(entry["tool_names"][0], "read_file");
        assert_eq!(entry["response_text_len"], 13);
        assert!(entry["usage"]["input_tokens"].as_u64() == Some(100));
        assert!(entry["usage"]["output_tokens"].as_u64() == Some(50));
        assert!(entry["duration_ms"].as_u64().is_some());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn maybe_profiled_plain_delegates() {
        let client = MaybeProfiled::Plain(MockLlm);
        let response = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response.text, "mock response");
    }
}
