//! Builtin tools for spawning and inspecting subagents (#112).
//!
//! This module is a TDD stub: the public surface (constants, the
//! generic `SubagentTools<C>` handle, `tool_definitions`, `supports_tool`,
//! and `execute_tool`) is in place so the test suite in
//! `tests/spawn_subagent.rs` compiles and runs, but the dispatch bodies
//! return errors. The next commit fills them in.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::inbound::ConversationService;

use crate::background_tasks::BackgroundTaskRegistry;

/// Tool name (LLM-visible) for spawning a subagent.
pub const TOOL_SPAWN_SUBAGENT: &str = "spawn_subagent";
/// Tool name (LLM-visible) for polling a previously-spawned subagent.
pub const TOOL_GET_SUBAGENT_STATUS: &str = "get_subagent_status";

/// Generic-over-`ConversationService` wrapper that publishes the two
/// builtin tools and dispatches them. Cheap to `Clone`.
pub struct SubagentTools<C: ConversationService> {
    #[allow(dead_code)]
    registry: Arc<BackgroundTaskRegistry>,
    #[allow(dead_code)]
    conversations: Arc<C>,
}

impl<C: ConversationService> Clone for SubagentTools<C> {
    fn clone(&self) -> Self {
        Self {
            registry: Arc::clone(&self.registry),
            conversations: Arc::clone(&self.conversations),
        }
    }
}

/// `true` when `name` is one of the tools defined here.
pub fn supports_tool(_name: &str) -> bool {
    false
}

/// Tool definitions advertised by this module (stub: empty until the
/// implementation commit).
pub fn tool_definitions() -> Vec<ToolDefinition> {
    Vec::new()
}

impl<C: ConversationService + 'static> SubagentTools<C> {
    pub fn new(registry: Arc<BackgroundTaskRegistry>, conversations: Arc<C>) -> Self {
        Self {
            registry,
            conversations,
        }
    }

    pub async fn execute_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        Err(CoreError::ToolExecution(
            "subagent tools not yet implemented".to_string(),
        ))
    }
}
