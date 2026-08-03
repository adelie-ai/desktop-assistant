use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use crate::CoreError;
use crate::domain::{Message, Role, ToolDefinition, ToolNamespace};
use crate::ports::llm::{
    ChunkCallback, HostedToolSearch, LlmClient, LlmResponse, ModelInfo, ModelListingReport,
    ReasoningConfig, TokenUsage, dispatch_namespaced,
};

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
                        // Char-boundary-safe cut (DA-2): a naive byte slice
                        // panics when byte 200 lands inside a multibyte char.
                        format!(
                            "{}...",
                            crate::planning::truncate_on_char_boundary(&m.content, 200)
                        )
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
                        // Char-boundary-safe cut (DA-2), as in log_messages.
                        format!(
                            "{}...",
                            crate::planning::truncate_on_char_boundary(&response.text, 200)
                        )
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

#[async_trait::async_trait]
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

    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        self.inner.list_models_detailed().await
    }

    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        self.inner.refresh_models_detailed().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
        let tool_count = tools.len();
        let message_count = messages.len();
        let profile_messages = self.profile_messages(&messages);

        let start = Instant::now();
        let result = self
            .inner
            .stream_completion(messages, tools, reasoning, on_chunk)
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

    /// Hands back `self`, never the inner client's object, so this
    /// decorator stays in the call path for a namespaced turn. See
    /// [`LlmClient::hosted_tool_search`].
    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        self.inner
            .hosted_tool_search()
            .is_some()
            .then_some(self as &dyn HostedToolSearch)
    }
}

#[async_trait::async_trait]
impl<L: LlmClient> HostedToolSearch for ProfilingLlmClient<L> {
    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
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
        let result = dispatch_namespaced(
            &self.inner,
            messages,
            core_tools,
            namespaces,
            reasoning,
            on_chunk,
        )
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
        if let Ok(env_path) = std::env::var("LLM_PROFILE_LOG")
            && !env_path.is_empty()
        {
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

        if !enabled {
            return Self::Plain(inner);
        }

        let resolve_tilde = |p: &str| -> PathBuf {
            if p.starts_with("~/")
                && let Ok(home) = std::env::var("HOME")
            {
                return PathBuf::from(home).join(&p[2..]);
            }
            PathBuf::from(p)
        };

        let path = log_path.map(resolve_tilde).unwrap_or_else(|| {
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

#[async_trait::async_trait]
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

    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        match self {
            Self::Plain(l) => l.list_models_detailed().await,
            Self::Profiled(l) => l.list_models_detailed().await,
        }
    }

    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        match self {
            Self::Plain(l) => l.refresh_models_detailed().await,
            Self::Profiled(l) => l.refresh_models_detailed().await,
        }
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match self {
            Self::Plain(l) => {
                l.stream_completion(messages, tools, reasoning, on_chunk)
                    .await
            }
            Self::Profiled(l) => {
                l.stream_completion(messages, tools, reasoning, on_chunk)
                    .await
            }
        }
    }

    /// Hands back the selected arm's object, not `self`, because this type
    /// is a transparent forwarder rather than a decorator - the same
    /// classification as the `Arc<T>` blanket impl.
    ///
    /// The rule in [`LlmClient::hosted_tool_search`] - a decorator returns
    /// `self` - exists so a decorator's own per-call work is not skipped on a
    /// namespaced turn. This enum has no per-call work: it only picks an arm.
    /// The `Profiled` arm hands back [`ProfilingLlmClient`]'s own object, so
    /// profiling stays in the path; the `Plain` arm hands back the inner
    /// client's. Both are the object the turn should reach, so inserting this
    /// enum between them would add a hop that can only be neutral.
    ///
    /// The one thing this must not do is answer `None` when an arm has hosted
    /// search, which would silently flatten every namespaced turn. That is
    /// what `maybe_profiled_forwards_the_hosted_search_object` pins.
    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        match self {
            Self::Plain(l) => l.hosted_tool_search(),
            Self::Profiled(l) => l.hosted_tool_search(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    struct MockLlm;

    #[async_trait::async_trait]
    impl LlmClient for MockLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
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
                ReasoningConfig::default(),
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

    /// Builds a string longer than 200 bytes whose byte 200 falls in the
    /// middle of a multibyte character, so a naive `&s[..200]` slice panics.
    fn multibyte_straddling_200() -> String {
        let mut s = "a".repeat(199);
        s.push_str(&"é".repeat(5));
        assert!(s.len() > 200);
        assert!(!s.is_char_boundary(200));
        s
    }

    #[tokio::test]
    async fn profiling_preview_handles_multibyte_message_content() {
        // DA-2: preview truncation of an inbound message must land on a char
        // boundary, not panic mid-character at byte 200.
        let dir = std::env::temp_dir().join(format!(
            "llm_profile_mb_msg_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("profile.jsonl");

        let client = ProfilingLlmClient::new(MockLlm, log_path, false);
        let response = client
            .stream_completion(
                vec![Message::new(Role::User, multibyte_straddling_200())],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response.text, "mock response");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn profiling_preview_handles_multibyte_response_text() {
        // DA-2: preview truncation of the LLM's response text must land on a
        // char boundary, not panic mid-character at byte 200.
        struct LongMultibyteLlm;

        #[async_trait::async_trait]
        impl LlmClient for LongMultibyteLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                Ok(LlmResponse::text(multibyte_straddling_200()))
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "llm_profile_mb_resp_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("profile.jsonl");

        let client = ProfilingLlmClient::new(LongMultibyteLlm, log_path, false);
        let response = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert!(response.text.starts_with("aaa"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn maybe_profiled_plain_delegates() {
        let client = MaybeProfiled::Plain(MockLlm);
        let response = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .unwrap();
        assert_eq!(response.text, "mock response");
    }

    /// Connector double that reports hosted tool search, paired with
    /// [`MockLlm`], which reports the trait default `false`. Every decorator
    /// below is asserted against both, because the two ways a decorator can
    /// answer for the wrong client fail in opposite directions:
    ///
    /// - Dropping the forward falls through to the trait default `false`,
    ///   which only the `HostedSearchLlm` case detects.
    /// - Hardcoding `true` invents a capability the inner client does not
    ///   have, which only the `MockLlm` case detects. This is the harmful
    ///   direction, and the shape of the defect this module's tests exist
    ///   for: a turn that believes in hosted search strips
    ///   `builtin_tool_search` and then sends the whole tool fleet to a
    ///   connector that cannot do hosted search.
    struct HostedSearchLlm;

    #[async_trait::async_trait]
    impl LlmClient for HostedSearchLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            Ok(LlmResponse::text("hosted"))
        }

        fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
            Some(self)
        }
    }

    #[async_trait::async_trait]
    impl HostedToolSearch for HostedSearchLlm {
        async fn stream_completion_with_namespaces(
            &self,
            _messages: Vec<Message>,
            _core_tools: &[ToolDefinition],
            _namespaces: &[ToolNamespace],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            Ok(LlmResponse::text("hosted namespaced"))
        }
    }

    /// Profiling log path for the capability tests. They never dispatch, so
    /// nothing is written; the path still goes under the temp directory so a
    /// later edit that does dispatch cannot drop a file in the working
    /// directory.
    ///
    /// `label` names the calling test. It cannot be `line!()` here: that
    /// macro expands where it is written, so a call inside this helper
    /// yields this line for every caller and the paths collide.
    fn capability_log_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "llm_profile_capability_{}_{label}.jsonl",
            std::process::id()
        ))
    }

    /// One test per decorator, so a mutation of one decorator names that
    /// decorator in the failure output rather than stopping the whole sweep
    /// at its first assertion.
    #[test]
    fn profiling_client_answers_hosted_tool_search_from_inner() {
        let path = capability_log_path("profiling");
        assert!(
            ProfilingLlmClient::new(HostedSearchLlm, path.clone(), false)
                .hosted_tool_search()
                .is_some(),
            "must forward the inner capability"
        );
        assert!(
            !ProfilingLlmClient::new(MockLlm, path, false)
                .hosted_tool_search()
                .is_some(),
            "must not invent a capability the inner client does not have"
        );
    }

    #[test]
    fn maybe_profiled_plain_answers_hosted_tool_search_from_inner() {
        assert!(
            MaybeProfiled::Plain(HostedSearchLlm)
                .hosted_tool_search()
                .is_some(),
            "must forward the inner capability"
        );
        assert!(
            !MaybeProfiled::Plain(MockLlm).hosted_tool_search().is_some(),
            "must not invent a capability the inner client does not have"
        );
    }

    #[test]
    fn maybe_profiled_profiled_answers_hosted_tool_search_from_inner() {
        let path = capability_log_path("maybe_profiled");
        assert!(
            MaybeProfiled::Profiled(ProfilingLlmClient::new(
                HostedSearchLlm,
                path.clone(),
                false
            ))
            .hosted_tool_search()
            .is_some(),
            "must forward the inner capability"
        );
        assert!(
            !MaybeProfiled::Profiled(ProfilingLlmClient::new(MockLlm, path, false))
                .hosted_tool_search()
                .is_some(),
            "must not invent a capability the inner client does not have"
        );
    }

    /// This wrapper sits around every client the daemon builds, so a wrong
    /// answer here reaches the whole fleet at once.
    #[test]
    fn retrying_client_answers_hosted_tool_search_from_inner() {
        use crate::ports::llm::RetryingLlmClient;

        assert!(
            RetryingLlmClient::new(HostedSearchLlm, 1)
                .hosted_tool_search()
                .is_some(),
            "must forward the inner capability"
        );
        assert!(
            !RetryingLlmClient::new(MockLlm, 1)
                .hosted_tool_search()
                .is_some(),
            "must not invent a capability the inner client does not have"
        );
    }

    // --- Decorators must stay in the path for a namespaced turn (#1033) ---
    //
    // Answering the capability correctly is not enough. A decorator that
    // reports hosted tool search and then hands the caller its *inner*
    // client's dispatch object is skipped for exactly the turns that carry
    // the most tools. `profiling_decorator_stays_in_the_namespaced_path`
    // observes the decorator's own effect on a namespaced turn, so that
    // bypass fails it.
    //
    // `MaybeProfiled` is deliberately not in that group. It is a transparent
    // forwarder with no per-call work, so it hands back the selected arm's
    // object and there is nothing to lose by not being in the path. Its test
    // pins the failure it *can* have: hiding an arm's hosted search.

    /// Profiling log path for the in-path tests, which do dispatch and so do
    /// write a file. Same shape as [`capability_log_path`]; kept separate so
    /// a reader is not misled by that helper's "never dispatch" contract.
    fn dispatch_log_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "llm_profile_dispatch_{}_{label}.jsonl",
            std::process::id()
        ))
    }

    #[tokio::test]
    async fn profiling_decorator_stays_in_the_namespaced_path() {
        use crate::ports::llm::dispatch_namespaced;
        use crate::ports::llm::hosted_search_test_support::*;

        let path = dispatch_log_path("profiling_in_path");
        let _ = std::fs::remove_file(&path);
        let inner = ProbeLlm::new(true);
        let probe = std::sync::Arc::clone(&inner.probe);
        let client = ProfilingLlmClient::new(inner, path.clone(), false);

        dispatch_namespaced(
            &client,
            vec![],
            &[],
            &[namespace("ns", vec![tool("deferred")])],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await
        .expect("probe turn");

        assert_eq!(
            probe.namespaced_calls(),
            1,
            "the turn reached the inner client's hosted dispatch"
        );
        let logged = std::fs::read_to_string(&path).unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        assert!(
            logged.contains("deferred"),
            "the profiling decorator must profile a namespaced turn, and it \
             cannot if the caller was handed the inner client's dispatch \
             object. Log was: {logged:?}"
        );
    }

    /// `MaybeProfiled` must not hide an arm's hosted tool search.
    ///
    /// Not an in-the-path test, and it could not be one: this enum has no
    /// per-call work, so handing back an arm's object rather than `self`
    /// loses nothing and no assertion could tell the two apart. What it can
    /// get wrong is answering `None` while an arm has hosted search, which
    /// silently flattens every namespaced turn - the whole tool fleet in one
    /// request with no discovery tool. Both arms are checked, because the
    /// enum answers them separately.
    #[tokio::test]
    async fn maybe_profiled_forwards_the_hosted_search_object() {
        use crate::ports::llm::dispatch_namespaced;
        use crate::ports::llm::hosted_search_test_support::*;

        // Profiled arm: the turn reaches hosted dispatch, and the profiling
        // that arm carries is still applied.
        let path = dispatch_log_path("maybe_profiled_forwarding");
        let _ = std::fs::remove_file(&path);
        let inner = ProbeLlm::new(true);
        let probe = std::sync::Arc::clone(&inner.probe);
        let profiled = MaybeProfiled::Profiled(ProfilingLlmClient::new(inner, path.clone(), false));
        assert!(
            profiled.hosted_tool_search().is_some(),
            "the Profiled arm's hosted search must not be hidden"
        );
        dispatch_namespaced(
            &profiled,
            vec![],
            &[],
            &[namespace("ns", vec![tool("deferred")])],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await
        .expect("probe turn");
        let logged = std::fs::read_to_string(&path).unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            probe.namespaced_calls(),
            1,
            "the Profiled arm must reach hosted dispatch, not flatten"
        );
        assert!(
            logged.contains("deferred"),
            "the Profiled arm carries a ProfilingLlmClient, whose profiling must \
             still apply to a namespaced turn. Log was: {logged:?}"
        );

        // Plain arm.
        let inner = ProbeLlm::new(true);
        let probe = std::sync::Arc::clone(&inner.probe);
        let plain = MaybeProfiled::Plain(inner);
        assert!(
            plain.hosted_tool_search().is_some(),
            "the Plain arm's hosted search must not be hidden"
        );
        dispatch_namespaced(
            &plain,
            vec![],
            &[],
            &[namespace("ns", vec![tool("deferred")])],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await
        .expect("probe turn");
        assert_eq!(
            probe.namespaced_calls(),
            1,
            "the Plain arm must pass a namespaced turn through to hosted dispatch"
        );
        assert_eq!(probe.plain_calls(), 0, "the turn never flattened");

        // A client with no hosted search must still answer `None`, so the
        // forward cannot be a hardcoded `Some`.
        assert!(
            MaybeProfiled::Plain(ProbeLlm::new(false))
                .hosted_tool_search()
                .is_none(),
            "must not invent a capability the arm does not have"
        );
    }

    #[test]
    fn arc_blanket_impl_answers_hosted_tool_search_from_inner() {
        use std::sync::Arc;

        let hosted: Arc<dyn LlmClient> = Arc::new(HostedSearchLlm);
        assert!(
            hosted.hosted_tool_search().is_some(),
            "must forward the inner capability"
        );
        let plain: Arc<dyn LlmClient> = Arc::new(MockLlm);
        assert!(
            !plain.hosted_tool_search().is_some(),
            "must not invent a capability the inner client does not have"
        );
    }
}
