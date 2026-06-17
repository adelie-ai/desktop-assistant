//! Builtin tools for spawning and inspecting subagents (#112).
//!
//! This module defines two LLM-facing tools that compose on top of the
//! `BackgroundTaskRegistry` (#111) and the [`spawn_agent_conversation`]
//! helper extracted by #113. Each one is a small wrapper:
//!
//! - [`TOOL_SPAWN_SUBAGENT`] — the parent LLM calls this to create a
//!   child conversation, run a fresh turn on it, and either block on
//!   the result (`wait=true`) or fire-and-forget (`wait=false`).
//! - [`TOOL_GET_SUBAGENT_STATUS`] — the parent can later poll a
//!   previously-spawned `wait=false` child by task id.
//!
//! ## Composition shape
//!
//! [`SubagentTools`] is generic over a [`ConversationService`] so the
//! daemon's full routing handler (model overrides, dispatch warnings,
//! conversation persistence) wires in without changes and tests pass a
//! lightweight fake. The body of `spawn_subagent`:
//!
//! 1. Reads the current user id (`current_user_id`) — subagents
//!    inherit the parent's identity (#105).
//! 2. Reads the current task id (`current_task_id`) — the parent task
//!    must already be registered with the registry (#111). When unset
//!    the tool refuses cleanly so misuses surface as recoverable tool
//!    errors instead of silently producing orphan tasks.
//! 3. Creates a fresh `Conversation` via `ConversationService` so the
//!    child has its own id and message history.
//! 4. Delegates the registry-spawn + run to [`spawn_agent_conversation`]
//!    with a `TaskKind::Subagent` kind factory. The helper threads the
//!    user identity, cancellation token, and (when provided) the tool
//!    allowlist through the run automatically.
//! 5. Appends a `ToolCall` log entry to the *parent* task carrying
//!    `{child_task_id, child_conversation_id}` so the UI can drill in.
//! 6. If `wait=true`, awaits the child via `registry.wait` (cancelling
//!    the child if the parent's cancellation token fires) and returns
//!    the child's final assistant text via the helper's result sink.
//!    If `wait=false`, returns a JSON object with the child's task id
//!    immediately.
//!
//! ## Out of scope (follow-ups)
//!
//! - **Dispatch-side allowlist enforcement**: the `tools` input is
//!   forwarded into the `TOOL_ALLOWLIST` task-local (#113 already
//!   installed the slot), but the LLM-tool-dispatch path that actually
//!   filters by it is a follow-up.
//! - **System prompt as a structured field**: the core `Conversation`
//!   type doesn't carry a separate system-prompt column; the requested
//!   `system_prompt` is prepended to the child's first user message so
//!   the LLM still receives the intent. Wiring it as a real
//!   `Role::System` message is a separate change.
//! - **Persistence of parent/child links across restarts**: #115/#129.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::current_cancellation_token;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use crate::background_tasks::{BackgroundTaskRegistry, current_task_id};
use crate::{AgentConversationSpec, AgentResultSink, spawn_agent_conversation};

/// Tool name (LLM-visible) for spawning a subagent.
pub const TOOL_SPAWN_SUBAGENT: &str = "spawn_subagent";
/// Tool name (LLM-visible) for polling a previously-spawned subagent.
pub const TOOL_GET_SUBAGENT_STATUS: &str = "get_subagent_status";

/// Maximum subagent recursion depth (issue #291). Depth is the number of
/// `Subagent` ancestors a task would have: a top-level conversation
/// spawning a subagent produces depth 1, that subagent spawning another
/// produces depth 2, and so on. A `spawn_subagent` whose resulting child
/// would exceed this cap is rejected with a recoverable error, bounding
/// recursive fan-out (a "restricted" subagent could otherwise spawn
/// subagents — including itself — without limit). 8 is generous enough for
/// legitimate multi-level delegation while preventing runaway recursion.
pub const MAX_SUBAGENT_DEPTH: usize = 8;

/// Generic-over-`ConversationService` wrapper that publishes the two
/// builtin tools and dispatches them. Cheap to `Clone` — only holds
/// `Arc`s.
pub struct SubagentTools<C: ConversationService> {
    registry: Arc<BackgroundTaskRegistry>,
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

/// `true` when `name` is one of the tools defined here. Free function
/// (rather than an associated function) so wrapping `ToolExecutor`
/// adapters can route by prefix without naming a `ConversationService`
/// type parameter. Mirrors the shape of `BuiltinToolService::supports_tool`
/// in `mcp-client`.
pub fn supports_tool(name: &str) -> bool {
    matches!(name, TOOL_SPAWN_SUBAGENT | TOOL_GET_SUBAGENT_STATUS)
}

/// Free function returning the tool definitions advertised by this
/// module. Use this when the caller doesn't have (or doesn't want to
/// name) a `ConversationService` type parameter — the daemon's startup
/// wiring is the prime example.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            TOOL_SPAWN_SUBAGENT,
            "Spawn a subagent: a child conversation that runs an LLM turn on a fresh \
             prompt and either returns its final answer (wait=true) or runs in the \
             background (wait=false). The child inherits your user identity and is \
             cancelled if you are cancelled. Use this to delegate sub-tasks that \
             need their own context (search, summarisation, multi-step research) \
             without polluting your own conversation history.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short human-friendly label for the subagent (shown in the process-manager UI)."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The first user message the subagent should respond to."
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt override. Defaults to a generic 'you are a helper for the parent task' template."
                    },
                    "connection": {
                        "type": "string",
                        "description": "Optional connection slug to route the subagent's LLM through."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model id override for the subagent's LLM."
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional allowlist of tool names the subagent may use. Defaults to the parent's full tool set."
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "When true (default), block until the subagent finishes and return its final answer. When false, return the subagent's task id immediately so the parent can poll later via get_subagent_status."
                    }
                },
                "required": ["name", "prompt"]
            }),
        ),
        ToolDefinition::new(
            TOOL_GET_SUBAGENT_STATUS,
            "Read the status of a previously-spawned subagent by task id. Returns \
             the current status (pending/running/completed/failed/cancelled), the \
             final assistant message if the subagent has completed, and an error \
             string if it failed.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The subagent's task id, as returned by a prior spawn_subagent { wait: false } call."
                    }
                },
                "required": ["task_id"]
            }),
        ),
    ]
}

impl<C: ConversationService + Send + Sync + 'static> SubagentTools<C> {
    /// Build a fresh `SubagentTools` over the given registry and
    /// conversation service. The daemon wires its full routing handler
    /// in; tests use a lightweight fake.
    pub fn new(registry: Arc<BackgroundTaskRegistry>, conversations: Arc<C>) -> Self {
        Self {
            registry,
            conversations,
        }
    }

    /// Dispatch one of the tools by name. The LLM's tool-call handler
    /// calls this exactly the way it calls `BuiltinToolService::execute_tool`;
    /// the daemon wires a wrapping `ToolExecutor` that delegates by name
    /// (out of scope for this slice — see `Out of scope` in the module
    /// docs).
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        match name {
            TOOL_SPAWN_SUBAGENT => self.spawn(arguments).await,
            TOOL_GET_SUBAGENT_STATUS => self.status(arguments).await,
            other => Err(CoreError::ToolExecution(format!(
                "unknown subagent tool: {other}"
            ))),
        }
    }

    /// Count how many `Subagent` tasks are on the chain from `task_id` up
    /// to the root (inclusive of `task_id` itself when it is a subagent).
    /// This is the number of subagent levels already nested above the child
    /// a spawn would create. A defensive cap on the walk length guards
    /// against a corrupted/cyclic parent chain so this can never loop
    /// unboundedly — a cycle would itself be treated as "too deep".
    fn subagent_depth(
        &self,
        user_id: &desktop_assistant_auth_jwt::UserId,
        task_id: &api::TaskId,
    ) -> usize {
        let mut depth = 0usize;
        let mut current = Some(task_id.clone());
        // Walk at most MAX_SUBAGENT_DEPTH + 1 hops; past that we already know
        // the child would exceed the cap, so the exact count no longer matters.
        for _ in 0..=MAX_SUBAGENT_DEPTH {
            let Some(id) = current else { break };
            let Some(view) = self.registry.get(user_id, &id) else {
                break;
            };
            if matches!(view.kind, api::TaskKind::Subagent { .. }) {
                depth += 1;
            }
            current = view.parent;
        }
        depth
    }

    async fn spawn(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let name = required_string(&arguments, "name")?;
        let prompt = required_string(&arguments, "prompt")?;
        let system_prompt = optional_string(&arguments, "system_prompt");
        let connection = optional_string(&arguments, "connection");
        let model = optional_string(&arguments, "model");
        let tools_allowlist = optional_string_array(&arguments, "tools");
        let wait = arguments
            .get("wait")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        // Identity + parent linkage must both be present — refuse with
        // recoverable errors when misused (e.g. called outside a task
        // body) so the LLM sees a clean message instead of a panic.
        let user_id = current_user_id();
        let parent_task_id = current_task_id().ok_or_else(|| {
            CoreError::ToolExecution(
                "spawn_subagent must be called from inside a registered background task"
                    .to_string(),
            )
        })?;

        // Recursion depth limit (issue #291). Walk the parent chain to count
        // how many `Subagent` ancestors the about-to-be-spawned child would
        // have; the child's own depth is that count + 1. Reject the spawn when
        // it would exceed `MAX_SUBAGENT_DEPTH` so a subagent cannot recurse
        // (or self-spawn) unboundedly. Done before any side effects (no child
        // conversation is created) so a rejected spawn leaves no orphan state.
        let child_depth = self.subagent_depth(&user_id, &parent_task_id) + 1;
        if child_depth > MAX_SUBAGENT_DEPTH {
            return Err(CoreError::ToolExecution(format!(
                "spawn_subagent rejected: subagent recursion depth limit ({MAX_SUBAGENT_DEPTH}) \
                 reached. This subagent is already nested too deeply to spawn another. Complete \
                 the work directly instead of delegating further."
            )));
        }

        // Create the child conversation. The title doubles as the
        // subagent label so the UI can show it without destructuring
        // `TaskKind`.
        let child_conv = self
            .conversations
            .create_conversation(format!("Subagent: {name}"), vec![])
            .await?;
        let child_conversation_id = child_conv.id.0.clone();

        // Build the override the child's LLM call should use. We
        // forward `connection` + `model` from the tool args; effort is
        // intentionally not forwarded here because the issue doesn't
        // ask for it. Daemon-side resolution still applies its full
        // priority chain on top.
        let override_selection = build_override(connection, model);

        // Stash for the child's final text so the parent can pull it
        // out after `registry.wait`. The helper writes into the sink;
        // we read after the child terminates.
        let result_slot: AgentResultSink = Arc::new(AsyncMutex::new(None));

        // The child's first user message folds the requested
        // `system_prompt` into a prefix so the LLM still receives the
        // intent — see the module-level "Out of scope" note for the
        // proper-system-prompt follow-up.
        let initial_prompt = effective_prompt(prompt, system_prompt, &name);

        let parent_for_kind = parent_task_id.clone();
        let name_for_kind = name.clone();
        let spec = AgentConversationSpec {
            user_id: user_id.clone(),
            name: name.clone(),
            title: format!("Subagent: {name}"),
            initial_prompt,
            override_selection,
            tools: if tools_allowlist.is_empty() {
                None
            } else {
                Some(tools_allowlist)
            },
            conversation_id: child_conversation_id.clone(),
            result_sink: Some(Arc::clone(&result_slot)),
        };

        let child_task_id = spawn_agent_conversation(
            Arc::clone(&self.registry),
            Arc::clone(&self.conversations),
            spec,
            move |conv_id| api::TaskKind::Subagent {
                parent_task_id: parent_for_kind,
                conversation_id: conv_id,
                name: name_for_kind,
            },
        );

        // Append the spawn record to the parent's log so the UI's
        // tool-call view links to the child task and the child's
        // conversation panel.
        self.registry.append_log(
            &user_id,
            &parent_task_id,
            api::LogLevel::Info,
            api::LogCategory::ToolCall,
            format!("spawn_subagent: {name}"),
            Some(serde_json::json!({
                "tool": TOOL_SPAWN_SUBAGENT,
                "child_task_id": child_task_id.0.clone(),
                "child_conversation_id": child_conversation_id.clone(),
                "child_name": name,
            })),
        );

        if !wait {
            return Ok(serde_json::json!({
                "child_task_id": child_task_id.0,
                "child_conversation_id": child_conversation_id,
            })
            .to_string());
        }

        // Block on the child, propagating parent cancellation. The
        // parent's per-turn cancellation token (installed by
        // `with_cancellation_token` at the top of
        // `send_prompt_with_override`) is what we listen on here — when
        // the parent task is cancelled the token trips, we cancel the
        // child registry-side, then drain its `wait` so the registry
        // has time to mark it `Cancelled` before we return.
        let parent_token = current_cancellation_token().unwrap_or_default();
        self.wait_for_child(&user_id, &child_task_id, &parent_token)
            .await;

        let slot = result_slot.lock().await;
        match slot.as_ref() {
            Some(Ok(text)) => Ok(text.clone()),
            Some(Err(reason)) => Err(CoreError::ToolExecution(format!(
                "subagent failed: {reason}"
            ))),
            None => Err(CoreError::ToolExecution(
                "subagent produced no result (this should not happen)".to_string(),
            )),
        }
    }

    /// Await `child` or the parent's cancellation token, cancelling
    /// the child and draining if the parent trips first. Factored out
    /// so the spawn body reads top-down.
    async fn wait_for_child(
        &self,
        user_id: &desktop_assistant_auth_jwt::UserId,
        child: &api::TaskId,
        parent_token: &CancellationToken,
    ) {
        tokio::select! {
            biased;
            _ = parent_token.cancelled() => {
                // Parent cancelled — propagate to the child and wait
                // for the registry to record the terminal state so
                // recursive tests see all generations reach
                // `Cancelled` before assertions.
                let _ = self.registry.cancel(user_id, child);
                self.registry.wait(child).await;
            }
            _ = self.registry.wait(child) => {
                // Child finished on its own — done.
            }
        }
    }

    async fn status(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let task_id_str = required_string(&arguments, "task_id")?;
        let user_id = current_user_id();
        let task_id = api::TaskId(task_id_str.clone());

        let Some(view) = self.registry.get(&user_id, &task_id) else {
            // Existence-hiding: a task that doesn't exist and a task
            // that belongs to another user surface identically. #105's
            // contract — don't leak existence to a probing LLM.
            return Ok(serde_json::json!({
                "error": "not_found",
                "task_id": task_id_str,
            })
            .to_string());
        };

        let status_str = match view.status {
            api::TaskStatus::Pending => "pending",
            api::TaskStatus::Running => "running",
            api::TaskStatus::Completed => "completed",
            api::TaskStatus::Failed => "failed",
            api::TaskStatus::Cancelled => "cancelled",
        };

        // For completed subagents we surface the recorded last error
        // (if any). The child's final assistant text is captured at
        // spawn time via the helper's result sink and returned through
        // `spawn_subagent { wait: true }`; the registry doesn't store
        // it out-of-band so `status` is intentionally lifecycle-only.
        let mut payload = serde_json::json!({
            "task_id": task_id_str,
            "status": status_str,
        });
        if let Some(err) = view.last_error.as_ref() {
            payload["error"] = serde_json::Value::String(err.clone());
        }
        Ok(payload.to_string())
    }
}

fn build_override(
    connection: Option<String>,
    model: Option<String>,
) -> Option<api::SendPromptOverride> {
    match (connection, model) {
        // Both fields are required for the override to be meaningful
        // to the daemon's resolution chain; partial fills fall back to
        // the conversation's stored / purpose-default selection.
        (Some(connection_id), Some(model_id)) => Some(api::SendPromptOverride {
            connection_id,
            model_id,
            effort: None,
        }),
        _ => None,
    }
}

fn effective_prompt(prompt: String, system_prompt: Option<String>, name: &str) -> String {
    // Until the core `Conversation` model carries a real system prompt
    // field, we fold the requested `system_prompt` into the prompt
    // itself so the child LLM still receives the intent. The default
    // template references the parent's tool-arg `name` so the
    // subagent knows its role.
    let prefix = system_prompt.unwrap_or_else(|| {
        format!(
            "You are a helper subagent named '{name}'. Carry out the parent task's \
             request crisply and return your final answer as plain text."
        )
    });
    format!("{prefix}\n\n{prompt}")
}

fn required_string(args: &serde_json::Value, key: &str) -> Result<String, CoreError> {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::ToolExecution(format!("missing required string argument: {key}")))
}

fn optional_string(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn optional_string_array(args: &serde_json::Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}
