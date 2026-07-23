//! Tool-executor wrapper that makes the LLM-facing subagent tools reachable
//! (#134 / #287 slice 7).
//!
//! [`SubagentAwareToolExecutor`] decorates any [`ToolExecutor`] (in the daemon,
//! the `McpToolExecutor`): it advertises `spawn_subagent` / `get_subagent_status`
//! in `core_tools`, routes those two names to [`SubagentTools`], and delegates
//! everything else to the inner executor.
//!
//! ## Breaking the executor <-> conversation cycle
//!
//! [`SubagentTools`] needs an `Arc<dyn ConversationService>` to create and run
//! child conversations, but that service (the daemon's routing handler) *owns*
//! this executor (the handler holds `tools: T`). Constructing the executor with
//! a strong `Arc` to the handler would form a reference cycle that leaks the
//! entire graph for the process lifetime; and the handler is built *after* this
//! executor is moved into it, so the `Arc` does not even exist yet at
//! construction time. Both problems are solved by a **late-set `Weak` slot**:
//! the executor holds `Arc<OnceLock<Weak<dyn ConversationService>>>`, empty at
//! construction; the daemon `set`s a `Weak` downgrade of the routing handler
//! the instant it exists. `execute_tool` upgrades the `Weak` per call. Storing
//! a `Weak` (not `Arc`) keeps the cycle non-owning, so nothing leaks.
//!
//! Two failure branches, both recoverable (never a panic): the slot is unset
//! (a construction-ordering bug — the daemon failed to wire it) or the upgrade
//! fails (the service has been dropped, e.g. at shutdown). They are logged
//! distinctly so a wiring regression is loud rather than masquerading as normal
//! shutdown behavior.

use std::sync::{Arc, OnceLock, Weak};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::tools::ToolExecutor;

use crate::background_tasks::BackgroundTaskRegistry;
use crate::subagent_tools::{self, SubagentTools};

/// A late-set slot carrying a `Weak` reference to the conversation service the
/// subagent tools dispatch through. Shared (`Arc`) between the daemon (which
/// sets it once the routing handler exists) and this executor (which reads it).
pub type ConversationSlot = Arc<OnceLock<Weak<dyn ConversationService>>>;

/// Wraps a [`ToolExecutor`], adding `spawn_subagent` / `get_subagent_status`.
///
/// Generic over the inner executor `T` so the `application` crate need not name
/// the daemon's concrete `McpToolExecutor`; the conversation is type-erased to
/// `dyn ConversationService` behind a `Weak` slot (see module docs).
pub struct SubagentAwareToolExecutor<T: ToolExecutor> {
    inner: T,
    registry: Arc<BackgroundTaskRegistry>,
    conversations: ConversationSlot,
    /// Session-scratchpad handles for the subagent result-handoff (#607/#608);
    /// threaded onto each `SubagentTools` built per dispatch. `None` leaves the
    /// pad round-trip inert.
    scratchpad: Option<crate::subagent_tools::SubagentScratchpad>,
}

impl<T: ToolExecutor> SubagentAwareToolExecutor<T> {
    /// Wrap `inner`. `conversations` starts empty; the daemon must `set` a
    /// `Weak` downgrade of the conversation service before any turn runs, or
    /// subagent tool calls fail closed with a recoverable error. `scratchpad`
    /// carries the session-pad result-handoff handles (#607/#608).
    pub fn new(
        inner: T,
        registry: Arc<BackgroundTaskRegistry>,
        conversations: ConversationSlot,
        scratchpad: Option<crate::subagent_tools::SubagentScratchpad>,
    ) -> Self {
        Self {
            inner,
            registry,
            conversations,
            scratchpad,
        }
    }

    /// Resolve the subagent-tools handle for one dispatch, or a recoverable
    /// error naming which wiring/lifecycle condition failed.
    fn resolve_subagent_tools(&self) -> Result<SubagentTools<dyn ConversationService>, CoreError> {
        match self.conversations.get() {
            None => {
                tracing::warn!(
                    "subagent executor conversation slot not yet wired; rejecting subagent tool call"
                );
                Err(CoreError::ToolExecution(
                    "subagent tools are not available yet (conversation service not wired)"
                        .to_string(),
                ))
            }
            Some(weak) => match weak.upgrade() {
                Some(conv) => {
                    let mut tools = SubagentTools::new(Arc::clone(&self.registry), conv);
                    if let Some(sp) = &self.scratchpad {
                        tools = tools.with_scratchpad(sp.clone());
                    }
                    Ok(tools)
                }
                None => {
                    tracing::warn!("conversation service dropped; rejecting subagent tool call");
                    Err(CoreError::ToolExecution(
                        "subagent tools are unavailable (conversation service shut down)"
                            .to_string(),
                    ))
                }
            },
        }
    }
}

impl<T: ToolExecutor> ToolExecutor for SubagentAwareToolExecutor<T> {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        let mut tools = self.inner.core_tools().await;
        tools.extend(subagent_tools::tool_definitions());
        tools
    }

    async fn search_tools(&self, query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        self.inner.search_tools(query).await
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        if subagent_tools::supports_tool(name) {
            return Ok(subagent_tools::tool_definitions()
                .into_iter()
                .find(|d| d.name == name));
        }
        self.inner.tool_definition(name).await
    }

    async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        self.inner.tool_namespaces().await
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        if subagent_tools::supports_tool(name) {
            return self
                .resolve_subagent_tools()?
                .execute_tool(name, arguments)
                .await;
        }
        self.inner.execute_tool(name, arguments).await
    }
}
