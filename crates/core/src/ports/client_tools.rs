//! Client-side tool execution port (#107 / #234).
//!
//! Rule #8 of `docs/architecture-evolution.md` says client-local MCPs
//! (file access, terminal, the user's own laptop tooling) execute on the
//! *client's* machine, not on the daemon. When the LLM picks one of those
//! tools the daemon must suspend the turn, emit a `client_tool_call`
//! event, and resume when the client posts the result back.
//!
//! The turn loop lives in [`crate::service`]; the coordinator that owns
//! the suspension state machine and the wire event lives one layer up in
//! `desktop-assistant-application` (it depends on the transport
//! `EventSink` and the turn-state store, which core does not know about).
//! To let the core loop *consult and suspend* without taking a dependency
//! on `application`, this module defines a thin outbound port —
//! [`ClientToolPort`] — that the application implements as an adapter
//! over its `ClientToolCoordinator`.
//!
//! ## Why a task-local rather than a handler field
//!
//! The port is per-*turn*: it closes over the turn's `task_id`, its
//! event sink, and the conversation id, which are only known once a
//! `send_prompt` is in flight. The [`crate::service::ConversationHandler`]
//! is constructed once at daemon start, so the port can't live on it.
//! This mirrors how the cancellation token, model override, and
//! system-prompt refinement are threaded (see [`crate::ports::llm`] and
//! [`crate::ports::auth`]): the application installs the per-turn adapter
//! via [`with_client_tools`] right before invoking the service, and the
//! dispatch loop reads it via [`current_client_tools`]. When the slot is
//! unset (single-tenant callers that registered no client tools, tests,
//! background workers) the loop behaves exactly as before — every tool is
//! server-side.

use std::sync::Arc;

use crate::CoreError;
use crate::domain::ToolDefinition;

/// Outbound port the turn loop uses to (a) learn which client-local tools
/// the current connection registered and (b) suspend the turn on one of
/// those calls, awaiting the client's result.
///
/// Scoped to the current user via the task-local `UserId` (the
/// implementing adapter reads `current_user_id()`), so registrations and
/// suspensions can never cross tenants.
///
/// Uses [`async_trait::async_trait`] so it is dyn-compatible — the
/// application installs a boxed adapter behind an `Arc<dyn ClientToolPort>`.
#[async_trait::async_trait]
pub trait ClientToolPort: Send + Sync {
    /// Tool definitions registered as client-local for the current user.
    /// These are merged into the tool set offered to the LLM for the turn
    /// so the model can actually pick them.
    async fn tool_definitions(&self) -> Vec<ToolDefinition>;

    /// True iff `name` is a client-local tool registered for the current
    /// user. The dispatch loop consults this before each tool execution:
    /// a match routes to [`ClientToolPort::execute`] (client-side
    /// suspension) instead of the server-side [`crate::ports::tools::ToolExecutor`].
    async fn is_registered(&self, name: &str) -> bool;

    /// Suspend the turn on a client-local tool call: emit the
    /// `client_tool_call` event and await the matching `client_tool_result`.
    ///
    /// Returns the result string the LLM should see as this tool's
    /// output, or a [`CoreError`] (e.g. `Cancelled` if the turn's
    /// cancellation token trips while suspended, or `ToolExecution` if the
    /// client reports a tool error). The caller threads the returned
    /// string back into the conversation exactly as it would a server-side
    /// tool result, so the LLM loop continues transparently.
    async fn execute(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError>;
}

tokio::task_local! {
    /// The per-turn client-tool adapter. Installed by the application's
    /// send-turn body via [`with_client_tools`] immediately before invoking
    /// the conversation service; read by the dispatch loop via
    /// [`current_client_tools`]. Unset for callers that never register
    /// client tools, so [`current_client_tools`] returns `None` and the
    /// loop keeps its pre-#234 server-side-only behaviour.
    static CLIENT_TOOLS: Arc<dyn ClientToolPort>;
}

/// Run `fut` with `port` installed as the current task-local client-tool
/// adapter. All [`current_client_tools`] calls inside the future (and any
/// sub-tasks that inherit the scope) observe `port`.
pub async fn with_client_tools<F, T>(port: Arc<dyn ClientToolPort>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CLIENT_TOOLS.scope(port, fut).await
}

/// The current task-local client-tool adapter, or `None` when no scope is
/// installed. Safe to call from any async context — never panics, never
/// blocks.
pub fn current_client_tools() -> Option<Arc<dyn ClientToolPort>> {
    CLIENT_TOOLS.try_with(Arc::clone).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakePort;

    #[async_trait::async_trait]
    impl ClientToolPort for FakePort {
        async fn tool_definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition::new(
                "fs_read",
                "read a file",
                serde_json::json!({}),
            )]
        }
        async fn is_registered(&self, name: &str) -> bool {
            name == "fs_read"
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            tool_name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            Ok(format!("client result from {tool_name}"))
        }
    }

    #[tokio::test]
    async fn current_client_tools_outside_scope_is_none() {
        assert!(current_client_tools().is_none());
    }

    #[tokio::test]
    async fn current_client_tools_inside_scope_returns_installed_port() {
        let port: Arc<dyn ClientToolPort> = Arc::new(FakePort);
        let (registered, defs_len) = with_client_tools(port, async {
            let p = current_client_tools().expect("port installed");
            (
                p.is_registered("fs_read").await,
                p.tool_definitions().await.len(),
            )
        })
        .await;
        assert!(registered);
        assert_eq!(defs_len, 1);
    }

    #[tokio::test]
    async fn spawned_task_outside_scope_sees_none() {
        // task_local slots do not cross `tokio::spawn`.
        let observed = tokio::spawn(async { current_client_tools().is_some() })
            .await
            .unwrap();
        assert!(!observed);
    }
}
