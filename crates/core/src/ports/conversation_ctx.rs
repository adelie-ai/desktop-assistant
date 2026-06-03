//! Request-scoped conversation context.
//!
//! This module exposes a task-local [`ConversationId`] slot so tool
//! executors can scope per-conversation side state (e.g. the scratchpad)
//! to the conversation currently being served — without the
//! [`crate::ports::tools::ToolExecutor::execute_tool`] signature growing a
//! `conversation_id` parameter that every tool and every test fixture would
//! have to thread.
//!
//! ## Why a task-local
//!
//! This mirrors the [`crate::ports::auth`] `UserId` precedent exactly. The
//! dispatch loop in [`crate::service`] knows the conversation id for the
//! whole turn; tool execution happens deep inside the MCP executor, which
//! has no conversation parameter. A task-local installed around the
//! `execute_tool` call carries the id to the builtin scratchpad tools
//! without changing the port API surface. See AGENTS.md ("cross-cutting
//! context propagates via `tokio::task_local!`").
//!
//! When the slot is unset (background workers, tests, any non-conversation
//! tool call), [`current_conversation_id`] returns `None`, and the
//! scratchpad tools surface a clear "requires an active conversation" error
//! rather than silently operating on the wrong store.

use crate::domain::ConversationId;

tokio::task_local! {
    /// The conversation being served for the current turn. Installed by the
    /// service dispatch loop via [`with_conversation_id`] around each tool
    /// execution; read by the builtin scratchpad tools via
    /// [`current_conversation_id`].
    static CONVERSATION_ID: ConversationId;
}

/// Run `fut` with `id` installed as the current task-local conversation.
/// All [`current_conversation_id`] calls inside the future observe `id`.
pub async fn with_conversation_id<F, T>(id: ConversationId, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CONVERSATION_ID.scope(id, fut).await
}

/// The current task-local conversation id, or `None` when no scope is
/// installed. Safe to call from any async context — never panics, never
/// blocks.
pub fn current_conversation_id() -> Option<ConversationId> {
    CONVERSATION_ID.try_with(|c| c.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn current_conversation_id_outside_scope_is_none() {
        assert!(current_conversation_id().is_none());
    }

    #[tokio::test]
    async fn current_conversation_id_inside_scope_returns_installed_value() {
        let observed =
            with_conversation_id(ConversationId::from("conv-1"), async { current_conversation_id() })
                .await;
        assert_eq!(observed, Some(ConversationId::from("conv-1")));
    }

    #[tokio::test]
    async fn nested_scopes_override_then_restore() {
        let result = with_conversation_id(ConversationId::from("outer"), async {
            let inner =
                with_conversation_id(ConversationId::from("inner"), async { current_conversation_id() })
                    .await;
            let after = current_conversation_id();
            (inner, after)
        })
        .await;
        assert_eq!(result.0, Some(ConversationId::from("inner")));
        assert_eq!(result.1, Some(ConversationId::from("outer")));
    }

    #[tokio::test]
    async fn spawned_task_outside_scope_sees_none() {
        // `task_local` slots do not cross `tokio::spawn`; a tool executed in
        // a spawned task without re-installing the scope must see `None`.
        let observed = tokio::spawn(async { current_conversation_id() })
            .await
            .unwrap();
        assert!(observed.is_none());
    }
}
