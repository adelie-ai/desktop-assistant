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
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::client_tools::ClientToolPort;
use desktop_assistant_core::ports::llm::current_cancellation_token;
use desktop_assistant_core::ports::session::current_session_id;
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

/// Default cap on how long a turn waits for a client to answer a client-tool
/// call before the suspension gives up (#262). Generous on purpose: a
/// legitimately slow interactive client (one that needs the user to act) must
/// not be cut off — the point is only to bound an *indefinite* wedge when a
/// client can't fulfil the tool at all (e.g. a text client offered a tool it
/// never registered, the #260 failure mode). On expiry the suspension resolves
/// as a tool error so the LLM loop continues to a terminal state instead of
/// parking forever. Conservative default; the daemon can install a
/// config-driven value per turn via [`with_client_tool_timeout`].
pub const DEFAULT_CLIENT_TOOL_TIMEOUT: Duration = Duration::from_secs(120);

tokio::task_local! {
    /// Per-turn override for the client-tool suspension cap. The daemon can
    /// install a config-driven value around the turn body; when unset (the
    /// common case) [`current_client_tool_timeout`] falls back to
    /// [`DEFAULT_CLIENT_TOOL_TIMEOUT`]. Tests install a short value to exercise
    /// the expiry path without waiting.
    static CLIENT_TOOL_TIMEOUT: Duration;
}

/// Run `fut` with `timeout` as the client-tool suspension cap (#262).
pub async fn with_client_tool_timeout<F, T>(timeout: Duration, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CLIENT_TOOL_TIMEOUT.scope(timeout, fut).await
}

/// The active client-tool suspension cap, or [`DEFAULT_CLIENT_TOOL_TIMEOUT`]
/// when no scope is installed.
fn current_client_tool_timeout() -> Duration {
    CLIENT_TOOL_TIMEOUT
        .try_with(|d| *d)
        .unwrap_or(DEFAULT_CLIENT_TOOL_TIMEOUT)
}

/// Coordinator for the registration + suspension halves of the client-
/// tool dance. One instance is shared by the whole daemon; concurrent
/// `RegisterClientTools` and `ClientToolResult` calls are serialized by
/// internal mutexes around small `HashMap`s — there is no async work
/// inside the mutex critical sections, so blocking is bounded.
pub struct ClientToolCoordinator {
    /// Currently-registered client-local tools, keyed by [`RegistrationKey`]
    /// (the `(user_id, session_id)` of the registering connection) → tool name
    /// → full registration (description + input schema). The full registration
    /// (not just the name) is retained so the turn loop can offer the tool's
    /// schema to the LLM (#234).
    ///
    /// Keying on the **login session**, not just the user, is the #261 fix:
    /// each client connection registers its own tools, so the voice daemon's
    /// `say_this` is offered only on the voice session's turns and never leaks
    /// onto a text client's turn (two windows of one user have independent
    /// sets). The `user_id` component is retained in the key so cross-user
    /// isolation holds even in the unscoped fallback bucket, where every
    /// connection would otherwise share a single sentinel session id.
    registrations: Mutex<HashMap<RegistrationKey, HashMap<String, api::ClientToolRegistration>>>,
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

/// Registry key for a client-tool registration: `(user_id, session_id)`.
/// Scoping by the login session keeps two connections of the same user
/// independent (#261); the `user_id` component preserves cross-user
/// isolation in the unscoped fallback bucket. Both come from the
/// transport-installed task-locals.
type RegistrationKey = (String, String);

/// The `(user_id, session_id)` of the calling connection, read from the
/// request-scoped task-locals.
fn current_registration_key() -> RegistrationKey {
    (
        current_user_id().as_str().to_string(),
        current_session_id().as_str().to_string(),
    )
}

impl ClientToolCoordinator {
    pub fn new() -> Self {
        Self {
            registrations: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Replace the registered tool set for the *current connection* (the
    /// `(user_id, session_id)` read from the request-scoped task-locals).
    /// Idempotent under the same set; clears previously-registered tools on
    /// this session that aren't in the new set. A re-register on one session
    /// never touches another session's set, even for the same user (#261).
    ///
    /// Returns the count of tools accepted.
    pub async fn register(&self, tools: &[api::ClientToolRegistration]) -> u32 {
        let key = current_registration_key();
        let mut regs = self.registrations.lock().unwrap();
        let entry = regs.entry(key).or_default();
        entry.clear();
        for t in tools {
            entry.insert(t.name.clone(), t.clone());
        }
        u32::try_from(entry.len()).unwrap_or(u32::MAX)
    }

    /// True iff `name` is registered for the current connection's session.
    pub async fn is_client_registered(&self, name: &str) -> bool {
        let key = current_registration_key();
        let regs = self.registrations.lock().unwrap();
        regs.get(&key)
            .map(|set| set.contains_key(name))
            .unwrap_or(false)
    }

    /// Test/diagnostic helper: true iff `name` is registered in **any**
    /// `(user, session)` bucket. Production code must use
    /// [`is_client_registered`], which is correctly scoped to the calling
    /// connection's session (#261). This exists for out-of-band callers
    /// (integration tests, diagnostics) that need to assert a registration
    /// landed without knowing the server-minted `session_id` of the
    /// connection that registered it — a session id that is never visible to
    /// a caller outside that connection's request scope.
    pub async fn is_registered_in_any_session(&self, name: &str) -> bool {
        let regs = self.registrations.lock().unwrap();
        regs.values().any(|set| set.contains_key(name))
    }

    /// The tool definitions registered as client-local for the current
    /// connection's session, in the shape the LLM tool list expects (#234).
    /// Maps each [`api::ClientToolRegistration`] to a core [`ToolDefinition`]
    /// so the turn loop can offer them to the model without `core` depending
    /// on `api-model`. A turn only ever sees the tools registered by the
    /// connection driving it (#261).
    pub async fn registered_definitions(&self) -> Vec<ToolDefinition> {
        let key = current_registration_key();
        let regs = self.registrations.lock().unwrap();
        regs.get(&key)
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

    /// Drop every registration belonging to the current session (read from
    /// the `session_id` task-local). Called when a connection closes so a
    /// long-lived daemon doesn't accumulate stale per-session buckets across
    /// reconnects (#261). A session belongs to exactly one user, so this
    /// normally removes a single bucket. In-flight suspensions (keyed by
    /// `task_id`) are intentionally left untouched — a closing connection may
    /// still have its `ClientToolResult` in flight on another path.
    pub fn clear_session(&self) {
        let session = current_session_id().as_str().to_string();
        let mut regs = self.registrations.lock().unwrap();
        regs.retain(|(_user, sess), _| sess != &session);
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
        _ = tokio::time::sleep(current_client_tool_timeout()) => {
            // The client never answered (#262). Don't wedge the turn: drop the
            // slot so a late `ClientToolResult` is cleanly rejected
            // (`TurnNotFound`), and fall through with a tool-error outcome so
            // the LLM loop sees a failed tool call and continues to a terminal
            // state rather than parking forever.
            coord.pending.lock().unwrap().remove(&task_id.0);
            Err(format!(
                "client did not respond to tool call '{}' within {}s",
                pending_call.tool_name,
                current_client_tool_timeout().as_secs()
            ))
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

    /// Number of rows currently retained. Test-only: lets the eviction test
    /// assert that terminal rows don't accumulate over the daemon's lifetime.
    #[cfg(test)]
    fn row_count(&self) -> usize {
        self.rows.lock().unwrap().len()
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
        // A turn driven to a terminal state is done: nothing in this process
        // reads its row again (the suspend/resolve dance has resolved, and the
        // startup sweep only cares about *non*-terminal rows). Evict it so the
        // map doesn't grow unbounded over the daemon's lifetime — the mirror of
        // the background-task eviction in #158 (DA-14 / #300).
        if status.is_terminal() {
            rows.remove(id);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn turn_row(id: &str, status: TurnStatus) -> TurnRow {
        TurnRow {
            id: id.to_string(),
            user_id: "u".to_string(),
            conversation_id: "c".to_string(),
            status,
            state: TurnStateJson::default(),
            last_error: None,
        }
    }

    #[tokio::test]
    async fn terminal_transition_evicts_the_turn_row() {
        // DA-14 (#300): a turn driven to a terminal status must not linger in
        // the in-memory map — otherwise the store grows unbounded over the
        // daemon's lifetime (mirror of the background-task eviction in #158).
        let store = InMemoryTurnStateStore::new();
        store
            .create_turn(turn_row("t1", TurnStatus::PendingLlm))
            .await
            .unwrap();
        assert_eq!(store.row_count(), 1);

        store
            .update_turn("t1", TurnStatus::Complete, &TurnStateJson::default(), None)
            .await
            .unwrap();

        assert_eq!(
            store.row_count(),
            0,
            "a Complete turn must be evicted from the map"
        );
        assert!(
            store.get_turn("t1").await.unwrap().is_none(),
            "the evicted row is gone"
        );
    }

    #[tokio::test]
    async fn failed_transition_also_evicts() {
        let store = InMemoryTurnStateStore::new();
        store
            .create_turn(turn_row("t1", TurnStatus::PendingClientTool))
            .await
            .unwrap();
        store
            .update_turn(
                "t1",
                TurnStatus::Failed,
                &TurnStateJson::default(),
                Some("cancelled"),
            )
            .await
            .unwrap();
        assert_eq!(store.row_count(), 0, "a Failed turn must be evicted too");
    }

    #[tokio::test]
    async fn non_terminal_transition_keeps_the_row() {
        // A mid-turn transition (e.g. suspending on a client tool) must keep
        // the row so the suspend/resolve dance has its backing record.
        let store = InMemoryTurnStateStore::new();
        store
            .create_turn(turn_row("t1", TurnStatus::PendingLlm))
            .await
            .unwrap();
        store
            .update_turn(
                "t1",
                TurnStatus::PendingClientTool,
                &TurnStateJson::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(store.row_count(), 1, "a non-terminal turn row stays");
        let row = store.get_turn("t1").await.unwrap().unwrap();
        assert_eq!(row.status, TurnStatus::PendingClientTool);
    }

    #[tokio::test]
    async fn updating_a_missing_turn_still_errors() {
        // Eviction must not turn a genuinely-missing turn into a silent success.
        let store = InMemoryTurnStateStore::new();
        let err = store
            .update_turn(
                "ghost",
                TurnStatus::Complete,
                &TurnStateJson::default(),
                None,
            )
            .await;
        assert!(err.is_err(), "updating an unknown turn id must still error");
    }
}
