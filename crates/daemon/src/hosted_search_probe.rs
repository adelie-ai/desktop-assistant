//! Test doubles for the hosted-tool-search seam, shared by the daemon's
//! three `LlmClient` decorators.
//!
//! Purpose: let each decorator's own test module prove that the decorator
//! stays in the call path when a turn carries namespaces. A decorator that
//! hands back its inner client's [`HostedToolSearch`] object instead of its
//! own is bypassed for exactly the turns that carry the most tools, and
//! nothing else in the workspace notices.
//!
//! Non-goals: connector behaviour on the wire (`registry.rs` probes that
//! against a mock server) and the compile-time half of the seam (a client
//! cannot report hosted search without implementing it, which no runtime test
//! can observe).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, HostedToolSearch, LlmClient, LlmResponse, ReasoningConfig,
};

/// Leaf `LlmClient` double for namespaced turns.
///
/// Records which entry point each turn arrived through, what reasoning
/// config it carried, and how many times it was called. It can also return a
/// fixed opaque provider error, so a decorator that reclassifies errors can
/// be seen doing that on the namespaced path.
pub struct NamespaceProbe {
    hosted: bool,
    plain: AtomicUsize,
    namespaced: AtomicUsize,
    /// Opaque provider error every call returns, if set. Lets a decorator
    /// that reclassifies errors be seen doing so on the namespaced path.
    opaque_error: Option<String>,
    /// Reasoning config seen by the most recent call, whichever path.
    seen_reasoning: Mutex<Option<ReasoningConfig>>,
}

impl NamespaceProbe {
    /// A probe with hosted tool search implemented.
    pub fn hosted() -> Self {
        Self::build(true, None)
    }

    /// A probe without hosted tool search, so namespaced turns flatten.
    pub fn plain() -> Self {
        Self::build(false, None)
    }

    /// A hosted probe that always returns an opaque `CoreError::Llm`.
    pub fn hosted_opaque_error(detail: &str) -> Self {
        Self::build(true, Some(detail.to_string()))
    }

    fn build(hosted: bool, opaque_error: Option<String>) -> Self {
        Self {
            hosted,
            plain: AtomicUsize::new(0),
            namespaced: AtomicUsize::new(0),
            opaque_error,
            seen_reasoning: Mutex::new(None),
        }
    }

    /// Turns that arrived through [`LlmClient::stream_completion`].
    pub fn plain_calls(&self) -> usize {
        self.plain.load(Ordering::SeqCst)
    }

    /// Turns that arrived through the hosted-search dispatch.
    pub fn namespaced_calls(&self) -> usize {
        self.namespaced.load(Ordering::SeqCst)
    }

    /// Reasoning config the last turn carried, whichever path it took.
    pub fn seen_reasoning(&self) -> Option<ReasoningConfig> {
        *self.seen_reasoning.lock().expect("probe lock")
    }

    fn record(&self, reasoning: ReasoningConfig) -> Result<LlmResponse, CoreError> {
        *self.seen_reasoning.lock().expect("probe lock") = Some(reasoning);
        match &self.opaque_error {
            Some(detail) => Err(CoreError::Llm(detail.clone())),
            None => Ok(LlmResponse::text("probe")),
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for NamespaceProbe {
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        _on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        self.plain.fetch_add(1, Ordering::SeqCst);
        self.record(reasoning)
    }

    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        self.hosted.then_some(self as &dyn HostedToolSearch)
    }
}

#[async_trait::async_trait]
impl HostedToolSearch for NamespaceProbe {
    async fn stream_completion_with_namespaces(
        &self,
        _messages: Vec<Message>,
        _core_tools: &[ToolDefinition],
        _namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        _on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        self.namespaced.fetch_add(1, Ordering::SeqCst);
        self.record(reasoning)
    }
}

/// A tool definition with the given name and an empty object schema.
pub fn probe_tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, "probe tool", serde_json::json!({"type": "object"}))
}

/// A one-tool namespace, the smallest input that exercises the seam.
pub fn probe_namespace() -> ToolNamespace {
    ToolNamespace::new(
        "probe_ns",
        "probe namespace",
        vec![probe_tool("probe_deferred")],
    )
}

/// A chunk callback that accepts everything and keeps nothing.
pub fn noop_chunk() -> ChunkCallback {
    Box::new(|_| true)
}
