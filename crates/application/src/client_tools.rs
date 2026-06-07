//! Client-side execution of client-local MCP tools (#107).
//!
//! This module is the application-layer implementation of the turn state
//! machine described in `docs/architecture-evolution.md` rule #8 and the
//! Phase-2 plan. It composes two pieces:
//!
//! 1. **`ClientToolCoordinator`** — the in-memory store for per-user
//!    registered client-local tool names plus the in-flight suspensions
//!    (one `oneshot::Sender<Result<String, String>>` per pending tool
//!    call). It does NOT touch the database; it is the hot-path mutex
//!    that lets a turn body suspend on `await` and wake up when the
//!    client's `ClientToolResult` arrives.
//!
//! 2. **`TurnStateStore`** (in `desktop-assistant-core::ports::store`)
//!    — the durable record of the turn's status. The coordinator writes
//!    transitions into the DB so a crashed daemon can sweep abandoned
//!    rows on restart, and so external observers (audit, the future
//!    background-task registry from #111) can read the current state.
//!
//! ## Why a coordinator separate from the registry in #111
//!
//! Issue #111 introduces a process-wide `BackgroundTaskRegistry` that
//! tracks every spawned task. That registry is about *tasks* — the
//! tokio futures backing each user-initiated turn. This module is about
//! *turns* — the LLM-loop state that may suspend on a client-local
//! tool. The two responsibilities overlap (every turn IS a task) but
//! the failure modes differ: registry cancellation cancels the future;
//! a coordinator-managed suspension waits for an external client
//! response. Keeping them separate means #111 can land on its own
//! schedule and this PR doesn't take a hard dep on it.
//!
//! ## Cross-user safety
//!
//! Every entry point reads the current `user_id` from the task-local
//! installed by the transport handler (`with_user_id`). Registrations
//! and suspensions are scoped by that id. The DB-side turn row carries
//! `user_id NOT NULL`; resolution refuses to touch rows whose user_id
//! disagrees with the resolver's, exactly the
//! `turn_cross_user_isolation` test acceptance criterion.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::client_tools::ClientToolPort;
use desktop_assistant_core::ports::llm::current_cancellation_token;
use desktop_assistant_core::ports::store::{
    PendingClientToolCall, TurnRow, TurnStateJson, TurnStateStore, TurnStatus,
};
use thiserror::Error;
use tokio::sync::oneshot;

use crate::EventSink;

/// Errors specific to the client-tool coordination layer.
///
/// Kept separate from `CoreError` because resolution failures originate
/// at the transport boundary (a malformed `ClientToolResult`) and the
/// error surface only matters to the application layer's adapter; core
/// services don't observe these.
#[derive(Debug, Error)]
pub enum ClientToolResolutionError {
    /// The task_id named in `ClientToolResult` has no pending
    /// suspension, OR the row belongs to a different user_id. The two
    /// cases are merged so cross-user probes can't distinguish them
    /// (#105's "don't leak existence" rule applied to turn rows).
    #[error("no pending turn for task_id={task_id} under this user")]
    TurnNotFound { task_id: String },

    /// The `tool_call_id` in `ClientToolResult` doesn't match the
    /// `tool_call_id` recorded in the suspended turn's state. Likely a
    /// confused client; the daemon refuses rather than feeding the
    /// LLM a result it didn't ask for.
    #[error("tool_call_id mismatch for task_id={task_id}: expected={expected}, got={got}")]
    ToolCallIdMismatch {
        task_id: String,
        expected: String,
        got: String,
    },

    /// The result payload is malformed (e.g. exceeds the configured
    /// size cap, or contains neither `result` nor `error`). The
    /// daemon-side validator emits this so transports can surface a
    /// clean failure to the client.
    #[error("malformed client tool result: {0}")]
    MalformedResult(String),

    /// The underlying storage call failed. Propagated from the
    /// `TurnStateStore` impl unchanged.
    #[error("storage error: {0}")]
    Storage(String),
}

impl From<CoreError> for ClientToolResolutionError {
    fn from(e: CoreError) -> Self {
        Self::Storage(e.to_string())
    }
}

/// Maximum body size (in bytes) for a single `ClientToolResult`. Larger
/// payloads are refused at the transport boundary to keep the LLM
/// prompt within the model's context window — the LLM sees the
/// rejection as a tool error.
const MAX_CLIENT_TOOL_RESULT_BYTES: usize = 1_048_576;

/// Coordinator for the registration + suspension halves of the client-
/// tool dance. One instance is shared by the whole daemon; concurrent
/// `RegisterClientTools` and `ClientToolResult` calls are serialized by
/// internal mutexes around small `HashMap`s — there is no async work
/// inside the mutex critical sections, so blocking is bounded.
pub struct ClientToolCoordinator {
    /// Per-user map of currently-registered client-local tools, keyed by
    /// tool name → full registration (description + input schema). The
    /// architecture's "per-session" registration semantic collapses to
    /// "per-user" in this slice: the application layer has no native
    /// concept of a connection session yet, and the tests pin that
    /// registration is overwritten on each new `RegisterClientTools`
    /// call so a reconnecting client gets the same per-session
    /// behaviour without us tracking the connection. The full registration
    /// (not just the name) is retained so the turn loop can offer the tool's
    /// schema to the LLM (#234).
    registrations: Mutex<HashMap<String, HashMap<String, api::ClientToolRegistration>>>,
    /// In-flight suspensions, keyed by task_id. Each entry holds the
    /// expected `tool_call_id` so the resolver can refuse mismatches
    /// without consulting the DB, and the oneshot sender used to wake
    /// the suspended turn body.
    pending: Mutex<HashMap<String, PendingSlot>>,
}

struct PendingSlot {
    user_id: String,
    expected_tool_call_id: String,
    waker: oneshot::Sender<Result<String, String>>,
}

impl ClientToolCoordinator {
    pub fn new() -> Self {
        Self {
            registrations: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Replace the registered tool set for the *current* user (read via
    /// `current_user_id()`). Idempotent under the same set; clears
    /// previously-registered tools that aren't in the new set.
    ///
    /// Returns the count of tools accepted.
    pub async fn register(&self, tools: &[api::ClientToolRegistration]) -> u32 {
        let user_id = current_user_id().as_str().to_string();
        let mut regs = self.registrations.lock().unwrap();
        let entry = regs.entry(user_id).or_default();
        entry.clear();
        for t in tools {
            entry.insert(t.name.clone(), t.clone());
        }
        u32::try_from(entry.len()).unwrap_or(u32::MAX)
    }

    /// True iff `name` is registered for the current user.
    pub async fn is_client_registered(&self, name: &str) -> bool {
        let user_id = current_user_id().as_str().to_string();
        let regs = self.registrations.lock().unwrap();
        regs.get(&user_id)
            .map(|set| set.contains_key(name))
            .unwrap_or(false)
    }

    /// The tool definitions registered as client-local for the current user,
    /// in the shape the LLM tool list expects (#234). Maps each
    /// [`api::ClientToolRegistration`] to a core [`ToolDefinition`] so the
    /// turn loop can offer them to the model without `core` depending on
    /// `api-model`.
    pub async fn registered_definitions(&self) -> Vec<ToolDefinition> {
        let user_id = current_user_id().as_str().to_string();
        let regs = self.registrations.lock().unwrap();
        regs.get(&user_id)
            .map(|set| {
                set.values()
                    .map(|r| {
                        ToolDefinition::new(
                            r.name.clone(),
                            r.description.clone(),
                            r.input_schema.clone(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for ClientToolCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level helper: register tools for the current user. Wraps
/// `ClientToolCoordinator::register` so call sites don't need to know
/// about the inner method names. Returns the count of tools accepted.
pub async fn register_client_tools(
    coord: &ClientToolCoordinator,
    tools: &[api::ClientToolRegistration],
) -> u32 {
    coord.register(tools).await
}

/// Suspend the current turn on a client-side tool call.
///
/// Effects:
/// 1. Validates that `pending_call.tool_name` is registered for the
///    current user. Refuses with `CoreError::ToolExecution(...)` if not.
/// 2. Updates the turn row to `pending_client_tool` with the pending
///    call recorded in `state_json`.
/// 3. Emits `Event::ClientToolCall` on the sink.
/// 4. Installs a oneshot in the coordinator's `pending` map keyed on
///    `task_id`, and `.await`s it.
/// 5. On wake (resolved by `resolve_client_tool_result`):
///    - On `Ok(s)`: writes the row back to `pending_llm`, clears the
///      pending call from `state_json`, returns `Ok(s)`.
///    - On `Err(reason)`: writes `failed` with reason, returns
///      `Err(CoreError::ToolExecution(reason))`.
/// 6. If the per-turn cancellation token (`current_cancellation_token`)
///    trips while suspended, the suspension exits with
///    `CoreError::Cancelled` and the row is marked `failed("cancelled")`.
///
/// The caller is the wrapping `ToolExecutor`'s `execute_tool` method —
/// it sees the returned `String` as the tool's output and continues
/// the LLM loop transparently.
pub async fn suspend_for_client_tool(
    coord: &ClientToolCoordinator,
    store: &dyn TurnStateStore,
    sink: &dyn EventSink,
    task_id: api::TaskId,
    conversation_id: String,
    pending_call: PendingClientToolCall,
) -> Result<String, CoreError> {
    // Step 1: registration guard. Refuse before mutating any state.
    if !coord.is_client_registered(&pending_call.tool_name).await {
        return Err(CoreError::ToolExecution(format!(
            "tool '{}' is not registered as a client-local tool for this user",
            pending_call.tool_name
        )));
    }

    let user_id = current_user_id().as_str().to_string();

    // Step 2: write turn row to pending_client_tool with the call payload.
    let state = TurnStateJson {
        version: 1,
        pending_client_tool: Some(pending_call.clone()),
    };
    store
        .update_turn(&task_id.0, TurnStatus::PendingClientTool, &state, None)
        .await?;

    // Step 3: emit the wire event.
    let event = api::Event::ClientToolCall {
        task_id: task_id.clone(),
        conversation_id: conversation_id.clone(),
        tool_call_id: pending_call.tool_call_id.clone(),
        tool_name: pending_call.tool_name.clone(),
        arguments: pending_call.arguments.clone(),
    };
    sink.emit(event).await;

    // Step 4: install oneshot and await.
    let (tx, rx) = oneshot::channel::<Result<String, String>>();
    {
        let mut pending = coord.pending.lock().unwrap();
        pending.insert(
            task_id.0.clone(),
            PendingSlot {
                user_id: user_id.clone(),
                expected_tool_call_id: pending_call.tool_call_id.clone(),
                waker: tx,
            },
        );
    }

    // Watch the per-turn cancellation token alongside the oneshot so
    // cancellation surfaces while suspended. If no token is installed
    // (legacy callers, single-tenant tests), `cancelled()` futures
    // resolve on a default never-cancelled token.
    let token = current_cancellation_token().unwrap_or_default();
    let outcome: Result<String, String> = tokio::select! {
        result = rx => match result {
            Ok(payload) => payload,
            Err(_) => Err("coordinator dropped the suspension waker".to_string()),
        },
        _ = token.cancelled() => {
            // Remove the slot so a late-arriving result doesn't try to
            // wake a no-op oneshot.
            coord.pending.lock().unwrap().remove(&task_id.0);
            // Mark the row failed with a cancellation reason; ignore
            // any storage error so we always surface `Cancelled` to
            // the caller.
            let _ = store
                .update_turn(&task_id.0, TurnStatus::Failed, &state, Some("cancelled"))
                .await;
            return Err(CoreError::Cancelled);
        }
    };

    // Step 5: write back the post-resolution state.
    match outcome {
        Ok(payload) => {
            let cleared = TurnStateJson {
                version: 1,
                pending_client_tool: None,
            };
            store
                .update_turn(&task_id.0, TurnStatus::PendingLlm, &cleared, None)
                .await?;
            Ok(payload)
        }
        Err(reason) => {
            let _ = store
                .update_turn(&task_id.0, TurnStatus::Failed, &state, Some(&reason))
                .await;
            Err(CoreError::ToolExecution(reason))
        }
    }
}

/// Resolve a pending suspension for the current `user_id`. Drives the
/// validations in this order:
/// 1. Reject if the payload is malformed (too large, neither result
///    nor error populated).
/// 2. Look up the pending slot; refuse if missing OR `user_id` mismatch
///    (same opacity rule as cross-user conversation reads).
/// 3. Refuse on `tool_call_id` mismatch.
/// 4. Send the result through the oneshot. The suspended task picks it
///    up and continues the LLM loop.
pub async fn resolve_client_tool_result(
    coord: &ClientToolCoordinator,
    _store: &dyn TurnStateStore,
    user_id: UserId,
    task_id: api::TaskId,
    tool_call_id: String,
    payload: Result<String, String>,
) -> Result<(), ClientToolResolutionError> {
    // 1. Validate payload size.
    if let Ok(body) = &payload
        && body.len() > MAX_CLIENT_TOOL_RESULT_BYTES
    {
        return Err(ClientToolResolutionError::MalformedResult(format!(
            "result body exceeds size cap ({} > {} bytes)",
            body.len(),
            MAX_CLIENT_TOOL_RESULT_BYTES
        )));
    }
    if let Err(reason) = &payload
        && reason.len() > MAX_CLIENT_TOOL_RESULT_BYTES
    {
        return Err(ClientToolResolutionError::MalformedResult(format!(
            "error reason exceeds size cap ({} > {} bytes)",
            reason.len(),
            MAX_CLIENT_TOOL_RESULT_BYTES
        )));
    }

    // 2 & 3: look up slot, check user_id + tool_call_id, take the waker.
    let slot = {
        let mut pending = coord.pending.lock().unwrap();
        let slot =
            pending
                .get(&task_id.0)
                .ok_or_else(|| ClientToolResolutionError::TurnNotFound {
                    task_id: task_id.0.clone(),
                })?;
        if slot.user_id != user_id.as_str() {
            // Cross-user probe: claim the row doesn't exist.
            return Err(ClientToolResolutionError::TurnNotFound {
                task_id: task_id.0.clone(),
            });
        }
        if slot.expected_tool_call_id != tool_call_id {
            return Err(ClientToolResolutionError::ToolCallIdMismatch {
                task_id: task_id.0.clone(),
                expected: slot.expected_tool_call_id.clone(),
                got: tool_call_id.clone(),
            });
        }
        // OK to take the slot.
        pending.remove(&task_id.0).expect("just found above")
    };

    // 4. Wake the suspended task. If it has already been dropped (e.g.
    // cancelled in the meantime) we silently ignore; the cleanup path
    // already removed the row.
    let _ = slot.waker.send(payload);
    Ok(())
}

/// Cold-restart sweep: mark every non-terminal turn row `failed` with
/// `last_error = "daemon_restarted"`. Called once at daemon startup so
/// abandoned turns don't accumulate. Returns the count swept.
///
/// The sweep is intentionally NOT scoped to a `user_id`: it walks the
/// whole table on the assumption that a daemon restart is a system
/// event affecting every user equally. Storage adapters implement
/// `scan_non_terminal` without applying the `current_user_id()` filter
/// for the same reason.
pub async fn sweep_non_terminal_turns_on_startup(
    store: &dyn TurnStateStore,
) -> Result<u32, CoreError> {
    let rows = store.scan_non_terminal().await?;
    let mut count = 0u32;
    for row in rows {
        // Preserve the row's state_json so a follow-up audit can see
        // *what* was pending when the daemon died.
        store
            .update_turn(
                &row.id,
                TurnStatus::Failed,
                &row.state,
                Some("daemon_restarted"),
            )
            .await?;
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// Per-turn adapter implementing the core [`ClientToolPort`] (#234).
///
/// The shared [`ClientToolCoordinator`] and [`TurnStateStore`] live for the
/// life of the daemon; the `task_id`, `conversation_id`, and the turn's
/// [`EventSink`] are only known once a `send_prompt` is in flight. The
/// application's send-turn body constructs one of these per turn (cheap — all
/// `Arc` clones plus three owned strings) and installs it via
/// [`desktop_assistant_core::ports::client_tools::with_client_tools`] so the
/// core dispatch loop can consult the registered set and suspend on a
/// client-local tool call without `core` depending on `application`.
///
/// User scoping is implicit: every coordinator entry point reads
/// `current_user_id()` from the task-local the dispatcher installed, so this
/// adapter's `tool_definitions`/`is_registered`/`execute` are all scoped to
/// the connection's user.
pub struct CoordinatorClientToolPort {
    coord: Arc<ClientToolCoordinator>,
    store: Arc<dyn TurnStateStore>,
    sink: Arc<dyn EventSink>,
    task_id: api::TaskId,
    conversation_id: String,
}

impl CoordinatorClientToolPort {
    /// Build the per-turn adapter. `task_id` is the registry task id for the
    /// turn — the same id the client received in `SendMessageAck` and the id
    /// the emitted `ClientToolCall` / inbound `ClientToolResult` correlate on.
    pub fn new(
        coord: Arc<ClientToolCoordinator>,
        store: Arc<dyn TurnStateStore>,
        sink: Arc<dyn EventSink>,
        task_id: api::TaskId,
        conversation_id: String,
    ) -> Self {
        Self {
            coord,
            store,
            sink,
            task_id,
            conversation_id,
        }
    }

    /// Ensure a turn row exists before the coordinator's `update_turn` writes
    /// the `pending_client_tool` transition. The row is created lazily on the
    /// first client-tool suspension of the turn (turns that never call a
    /// client tool never create a row). A row created by an earlier suspension
    /// in the same turn — or by a concurrent racer — is fine: a duplicate-id
    /// `create_turn` error is swallowed, leaving the existing row untouched.
    async fn ensure_turn_row(&self) {
        let row = TurnRow {
            id: self.task_id.0.clone(),
            user_id: current_user_id().as_str().to_string(),
            conversation_id: self.conversation_id.clone(),
            status: TurnStatus::PendingLlm,
            state: TurnStateJson::default(),
            last_error: None,
        };
        // Best-effort: a dup id just means the row already exists. Any other
        // storage error is left for `suspend_for_client_tool`'s own
        // `update_turn` to surface.
        let _ = self.store.create_turn(row).await;
    }
}

/// In-memory [`TurnStateStore`] for the live single-node deploy (#234).
///
/// Phase-2 of the architecture (`docs/architecture-evolution.md`) calls for
/// a DB-persisted turn-state machine so a crashed daemon can sweep abandoned
/// turns and a Lambda invocation can resume one. That durable store is future
/// work; this slice activates client-tool execution on the live UDS path,
/// where the daemon is a long-lived single process and the turn row only
/// needs to survive within that process. This map provides exactly that: it
/// records the `pending_client_tool` transition so the coordinator's
/// suspend/resolve dance has a backing row, and `scan_non_terminal` lets the
/// startup sweep clear anything a restart left behind (here, nothing, since
/// the map starts empty). Swapping in a `PgTurnStateStore` later is a drop-in
/// behind the same trait.
#[derive(Default)]
pub struct InMemoryTurnStateStore {
    rows: Mutex<HashMap<String, TurnRow>>,
}

impl InMemoryTurnStateStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl TurnStateStore for InMemoryTurnStateStore {
    async fn create_turn(&self, row: TurnRow) -> Result<(), CoreError> {
        let mut rows = self.rows.lock().unwrap();
        if rows.contains_key(&row.id) {
            return Err(CoreError::Storage(format!("duplicate turn id: {}", row.id)));
        }
        rows.insert(row.id.clone(), row);
        Ok(())
    }

    async fn get_turn(&self, id: &str) -> Result<Option<TurnRow>, CoreError> {
        Ok(self.rows.lock().unwrap().get(id).cloned())
    }

    async fn update_turn(
        &self,
        id: &str,
        status: TurnStatus,
        state: &TurnStateJson,
        last_error: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut rows = self.rows.lock().unwrap();
        let row = rows
            .get_mut(id)
            .ok_or_else(|| CoreError::Storage(format!("turn missing: {id}")))?;
        row.status = status;
        row.state = state.clone();
        row.last_error = last_error.map(String::from);
        Ok(())
    }

    async fn scan_non_terminal(&self) -> Result<Vec<TurnRow>, CoreError> {
        Ok(self
            .rows
            .lock()
            .unwrap()
            .values()
            .filter(|r| !r.status.is_terminal())
            .cloned()
            .collect())
    }
}

#[async_trait::async_trait]
impl ClientToolPort for CoordinatorClientToolPort {
    async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.coord.registered_definitions().await
    }

    async fn is_registered(&self, name: &str) -> bool {
        self.coord.is_client_registered(name).await
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        self.ensure_turn_row().await;
        suspend_for_client_tool(
            &self.coord,
            &*self.store,
            &*self.sink,
            self.task_id.clone(),
            self.conversation_id.clone(),
            PendingClientToolCall {
                tool_call_id: tool_call_id.to_string(),
                tool_name: tool_name.to_string(),
                arguments,
            },
        )
        .await
    }
}
