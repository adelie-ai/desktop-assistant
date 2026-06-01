//! In-memory, user-scoped registry of background tasks (#111).
//!
//! `BackgroundTaskRegistry` is the unifying abstraction for every
//! cancellable, log-emitting unit of work the daemon runs on behalf of a
//! user. Today it backs three call sites (one wired in this PR, the
//! others in #112/#113):
//!
//! - **Foreground send turns** — `handle_send_message_with_override`
//!   spawns through the registry so the in-flight turn shows up in the
//!   process-manager UI and so the user's Cancel button has something
//!   concrete to trip (#109's [`CancellationToken`] is the wire).
//! - **Subagent invocations** (#112) — the parent task's
//!   `spawn_subagent` tool spawns a child task; the parent's body
//!   awaits the child by polling the registry.
//! - **Standalone agents** (#113) — top-level user-launched runs that
//!   have no waiting parent.
//!
//! ## Persistence is out of scope
//!
//! This module is intentionally in-memory only. Durability lands in
//! #115 once #107's DB-persisted state machine ships and we have
//! somewhere to anchor the resume logic. Keeping the in-memory shape
//! minimal now means the persistent variant can layer over the same
//! public API without churning callers.
//!
//! ## Concurrency model
//!
//! - A single [`Mutex`] (`std::sync::Mutex`) guards both the task map
//!   and the per-user broadcast-sender map. The registry is exclusively
//!   on-CPU work (HashMap ops, log appends, status flips) — short
//!   critical sections, no `.await` while holding the lock.
//! - Each task gets its own `tokio::task::spawn`. Cancellation is
//!   cooperative: the registry trips the [`CancellationToken`]; the
//!   task body is responsible for noticing.
//! - Events fan out via a per-user [`broadcast::Sender`]. Slow
//!   subscribers drop oldest events (standard broadcast semantics) —
//!   we don't apply back-pressure to the producing task.
//!
//! ## User scoping
//!
//! Every public operation takes `user_id`. Cross-user `cancel`/`get`/
//! `logs` calls return [`TaskError::NotFound`] — never `Forbidden`,
//! because that would leak existence (#105 contract).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore,
};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing::{error, warn};

tokio::task_local! {
    /// The id of the currently-executing background task.
    ///
    /// Installed by [`BackgroundTaskRegistry::spawn`] around the task
    /// body so nested code (LLM tool dispatch, in particular) can read
    /// "what task am I running under?" without threading the id through
    /// every signature. This is the parent-link for `spawn_subagent`
    /// (#112): the child registers `TaskKind::Subagent { parent_task_id,
    /// ... }` by reading this value at tool-dispatch time.
    ///
    /// When unset — tests that don't go through `spawn`, dreaming jobs,
    /// any code that runs outside a registered task — [`current_task_id`]
    /// returns `None`. Callers that depend on the value must handle the
    /// absence path explicitly.
    static CURRENT_TASK_ID: api::TaskId;
}

/// Returns the id of the task currently executing, when the call site
/// is inside a [`BackgroundTaskRegistry::spawn`] body. Returns `None`
/// outside any registered task body — see [`CURRENT_TASK_ID`].
pub fn current_task_id() -> Option<api::TaskId> {
    CURRENT_TASK_ID.try_with(|id| id.clone()).ok()
}

/// Configuration knobs for the registry.
///
/// Daemon-level config can override these via [`BackgroundTaskRegistry::with_config`];
/// every field has a sensible default for tests and single-tenant deploys.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Maximum number of log entries retained per task. Older entries
    /// are evicted when the buffer is full. The retained log is always
    /// the *most recent* slice — `seq` numbers stay monotonic across
    /// evictions so paging via `after_seq` keeps working.
    pub log_ring_capacity: usize,
    /// Capacity of the per-user broadcast channel that fans `Event::Task*`
    /// out to subscribers. Slow receivers drop oldest events; this is the
    /// `tokio::sync::broadcast` contract.
    pub broadcast_capacity: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        // Defaults chosen to match the issue spec (1000 log entries) and
        // to keep small UI consumers happy for ~a few minutes of events.
        Self {
            log_ring_capacity: 1000,
            broadcast_capacity: 256,
        }
    }
}

/// Errors returned by registry operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TaskError {
    /// The id is unknown, or it exists but belongs to a different user.
    /// We deliberately do not distinguish these cases — leaking existence
    /// would break the user-isolation contract (#105).
    #[error("task not found")]
    NotFound,

    /// The task is already in a terminal state — typically a `Failed`
    /// row that survived a daemon restart. `cancel` on such a row would
    /// otherwise look like a silent no-op; surfacing this distinct
    /// variant lets transport adapters return a clean 409-style error
    /// to the client instead of pretending the cancel succeeded (#115).
    #[error("task is already terminal")]
    AlreadyTerminal,
}

/// Per-task event sink for log lines emitted by a running task body.
///
/// Cloning the sink is cheap and intentional — the task body can hand
/// clones to nested helpers (tool runners, model adapters) so each can
/// emit lifecycle logs without re-fetching context.
#[derive(Clone)]
pub struct TaskLogSink {
    inner: Arc<Inner>,
    task_id: api::TaskId,
}

impl TaskLogSink {
    /// Append a log line to the task's bounded ring buffer and broadcast
    /// a [`api::Event::TaskLogAppended`] to the user's subscribers.
    ///
    /// Silent no-op if the task no longer exists (e.g. it was already
    /// removed) — callers should never observe a logging error.
    pub fn append(
        &self,
        level: api::LogLevel,
        category: api::LogCategory,
        message: String,
        data: Option<serde_json::Value>,
    ) {
        let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let Some(state) = tasks.get_mut(&self.task_id) else {
            return;
        };
        let entry = api::TaskLogEntry {
            seq: state.next_seq,
            timestamp: now_ms(),
            level,
            category,
            message,
            data,
        };
        state.next_seq += 1;
        if state.logs.len() == self.inner.config.log_ring_capacity {
            state.logs.pop_front();
        }
        state.logs.push_back(entry.clone());
        let owner = state.owner.clone();
        let task_id = self.task_id.clone();
        drop(tasks);
        self.inner.broadcast(
            &owner,
            api::Event::TaskLogAppended {
                id: task_id.0,
                entry,
            },
        );
    }
}

/// Per-task context handed to the body closure.
///
/// Fields are public so the body can pattern-match or capture them
/// freely. The non-public `inner` handle is what `set_progress_hint`
/// uses to mutate task state; we keep it private so callers can't
/// reach in and corrupt the registry from inside a task.
#[derive(Clone)]
pub struct TaskContext {
    pub task_id: api::TaskId,
    pub user_id: UserId,
    pub parent: Option<api::TaskId>,
    pub token: CancellationToken,
    pub logs: TaskLogSink,
    inner: Arc<Inner>,
}

impl TaskContext {
    /// Update the task's `progress_hint`. Visible immediately to
    /// `list`/`get` and broadcast via [`api::Event::TaskProgress`].
    pub fn set_progress_hint(&self, hint: Option<String>) {
        let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let Some(state) = tasks.get_mut(&self.task_id) else {
            return;
        };
        state.view.progress_hint = hint.clone();
        let owner = state.owner.clone();
        let id = self.task_id.0.clone();
        drop(tasks);
        self.inner.broadcast(
            &owner,
            api::Event::TaskProgress {
                id,
                progress_hint: hint,
            },
        );
    }
}

/// In-memory, user-scoped task registry.
///
/// Cheap to `Clone` (it's `Arc`-wrapped internally) so a single
/// registry instance can be shared across the daemon's handler, the
/// transport adapters, and tests.
#[derive(Clone)]
pub struct BackgroundTaskRegistry {
    inner: Arc<Inner>,
}

impl Default for BackgroundTaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundTaskRegistry {
    /// Build a registry with default configuration.
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Build a registry with the supplied configuration.
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                tasks: Mutex::new(HashMap::new()),
                user_channels: Mutex::new(HashMap::new()),
                completion_notify: Mutex::new(HashMap::new()),
                store: None,
            }),
        }
    }

    /// Attach a [`BackgroundTaskStore`] so spawned rows are mirrored to
    /// the database and survive a daemon restart (#115).
    ///
    /// When no store is attached the registry behaves as a pure in-memory
    /// cache — every existing single-tenant test path takes this branch.
    /// When a store is attached, `spawn` writes a row before the body
    /// starts, finalize updates that row with the terminal status, and
    /// the daemon-startup hook `sweep_non_terminal_on_startup` marks
    /// every row left behind by a previous incarnation as `Failed`.
    ///
    /// The store is held behind a `dyn` trait object because the daemon's
    /// `main.rs` may want to test with an in-memory mock without
    /// monomorphizing the registry against the storage crate. The trait
    /// is `Send + Sync` so cloning the registry across tokio tasks
    /// works without extra boxing.
    pub fn with_store(mut self, store: Arc<dyn BackgroundTaskStore>) -> Self {
        // Mutate via Arc::get_mut: cheap, panics if any other clone
        // exists (which would be a programming error — the store is
        // attached at construction time, before any clone goes out).
        let inner =
            Arc::get_mut(&mut self.inner).expect("with_store called after the registry was cloned");
        inner.store = Some(store);
        self
    }

    /// Spawn a new background task under `user_id`.
    ///
    /// Returns the new task id immediately; the task body runs on the
    /// current tokio runtime. The body must observe `ctx.token` to be
    /// cooperatively cancellable — see the module docs.
    pub fn spawn<F, Fut>(
        &self,
        user_id: UserId,
        kind: api::TaskKind,
        title: String,
        body: F,
    ) -> api::TaskId
    where
        F: FnOnce(TaskContext) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        let task_id = api::TaskId(uuid::Uuid::new_v4().to_string());
        let parent = parent_task_id(&kind);
        let token = CancellationToken::new();

        let now = now_ms();
        let view = api::TaskView {
            id: task_id.clone(),
            kind,
            status: api::TaskStatus::Running,
            started_at: now,
            ended_at: None,
            last_error: None,
            parent: parent.clone(),
            children: Vec::new(),
            title,
            progress_hint: None,
        };

        // Insert state, register child-link with parent (if any), and
        // emit the TaskStarted event under the lock-free window so
        // subscribers see registration in order.
        {
            let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
            tasks.insert(
                task_id.clone(),
                TaskState {
                    owner: user_id.clone(),
                    view: view.clone(),
                    token: token.clone(),
                    logs: VecDeque::with_capacity(self.inner.config.log_ring_capacity.min(64)),
                    // Seq numbers start at 1 so callers can pass
                    // `after_seq=0` to mean "I've seen nothing yet" —
                    // the filter in [`BackgroundTaskRegistry::logs`] is
                    // strict-greater-than.
                    next_seq: 1,
                    completed: false,
                },
            );
            if let Some(parent_id) = &parent
                && let Some(parent_state) = tasks.get_mut(parent_id)
            {
                parent_state.view.children.push(task_id.clone());
            }
        }

        self.inner
            .broadcast(&user_id, api::Event::TaskStarted { task: view.clone() });

        let logs_sink = TaskLogSink {
            inner: Arc::clone(&self.inner),
            task_id: task_id.clone(),
        };
        let ctx = TaskContext {
            task_id: task_id.clone(),
            user_id: user_id.clone(),
            parent,
            token: token.clone(),
            logs: logs_sink,
            inner: Arc::clone(&self.inner),
        };

        // Lifecycle log so the UI can show a clean "started" marker
        // without needing to subscribe before the spawn.
        ctx.logs.append(
            api::LogLevel::Info,
            api::LogCategory::Lifecycle,
            "task started".into(),
            None,
        );

        let inner = Arc::clone(&self.inner);
        let task_id_for_body = task_id.clone();
        let user_id_for_body = user_id.clone();
        let ctx_for_body = ctx.clone();
        let view_for_persist = view.clone();
        let task_id_for_scope = task_id.clone();
        tokio::spawn(async move {
            // Mirror the in-memory row to the persistence store BEFORE
            // running the user body. The await happens inside the spawned
            // task (spawn() itself is non-async by design) so the registry's
            // public API stays unchanged. If the write fails we log and
            // keep going — losing durability is a strictly worse outcome
            // than refusing to run the work, so we degrade rather than
            // abort. The cold-restart sweep would only miss this row
            // anyway; the user-visible task continues to work.
            inner
                .persist_create(&task_id_for_body, &user_id_for_body, &view_for_persist)
                .await;

            // Run the body inside a `CURRENT_TASK_ID` task-local scope so
            // nested tool dispatches (the `spawn_subagent` builtin, in
            // particular) can read the running task's id without threading
            // it through every signature.
            //
            // The body runs in a *child* task so that a panic surfaces as a
            // `JoinError` we can finalize on, instead of unwinding this
            // finalizer task. If a panicking body skipped `finalize`, the
            // row would be stuck `Running` forever — never broadcast, never
            // evicted — and every `wait()` on it would hang (#171). A panic
            // is recorded as a terminal `Failed`.
            let body_task =
                tokio::spawn(CURRENT_TASK_ID.scope(task_id_for_scope, body(ctx_for_body)));
            let result = match body_task.await {
                Ok(outcome) => outcome,
                Err(join_err) if join_err.is_panic() => {
                    let payload = join_err.into_panic();
                    let msg = payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "task body panicked".to_string());
                    Err(anyhow::anyhow!("task body panicked: {msg}"))
                }
                Err(join_err) => Err(anyhow::anyhow!("task body aborted: {join_err}")),
            };
            inner
                .finalize(&task_id_for_body, &user_id_for_body, result)
                .await;
        });

        task_id
    }

    /// Request cancellation of `id` owned by `user_id`. Cooperative — the
    /// running future is responsible for noticing.
    ///
    /// Returns:
    /// - `Err(TaskError::NotFound)` when the id is unknown or owned by
    ///   another user (the #105 opacity rule conflates the two).
    /// - `Err(TaskError::AlreadyTerminal)` when the row exists and
    ///   belongs to the caller but is in a terminal state — typically a
    ///   row that survived a daemon restart and was marked `Failed` by
    ///   the cold-restart sweep. The distinct variant prevents silent
    ///   no-ops and lets transport adapters return a clean error to the
    ///   client (#115).
    /// - `Ok(())` after tripping the cancellation token; the task body
    ///   is responsible for noticing and yielding.
    pub fn cancel(&self, user_id: &UserId, id: &api::TaskId) -> Result<(), TaskError> {
        let tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let Some(state) = tasks.get(id) else {
            return Err(TaskError::NotFound);
        };
        if &state.owner != user_id {
            // Existence-hiding: pretend it doesn't exist (#105 contract).
            return Err(TaskError::NotFound);
        }
        if matches!(
            state.view.status,
            api::TaskStatus::Completed | api::TaskStatus::Failed | api::TaskStatus::Cancelled
        ) {
            return Err(TaskError::AlreadyTerminal);
        }
        state.token.cancel();
        Ok(())
    }

    /// List tasks owned by `user_id`. When `include_finished` is false
    /// only `Pending`/`Running` tasks are returned. `limit` caps the
    /// returned slice (most-recently-started first).
    pub fn list(
        &self,
        user_id: &UserId,
        include_finished: bool,
        limit: Option<u32>,
    ) -> Vec<api::TaskView> {
        let tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let mut out: Vec<api::TaskView> = tasks
            .values()
            .filter(|s| &s.owner == user_id)
            .filter(|s| {
                if include_finished {
                    true
                } else {
                    matches!(
                        s.view.status,
                        api::TaskStatus::Pending | api::TaskStatus::Running
                    )
                }
            })
            .map(|s| s.view.clone())
            .collect();
        // Sort newest-first for stable list ordering — the UI expects
        // most-recent at the top.
        out.sort_by_key(|view| std::cmp::Reverse(view.started_at));
        if let Some(limit) = limit {
            out.truncate(limit as usize);
        }
        out
    }

    /// Fetch a single task view; `None` when the id is unknown or
    /// owned by another user.
    pub fn get(&self, user_id: &UserId, id: &api::TaskId) -> Option<api::TaskView> {
        let tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let state = tasks.get(id)?;
        if &state.owner != user_id {
            return None;
        }
        Some(state.view.clone())
    }

    /// Page log entries for `id`. `after_seq` is exclusive: a value of
    /// `0` returns from the oldest *retained* entry (which may be
    /// `seq=N>0` once the ring has evicted older lines). Returns the
    /// next sequence number callers should pass to resume.
    pub fn logs(
        &self,
        user_id: &UserId,
        id: &api::TaskId,
        after_seq: u64,
        limit: u32,
    ) -> Result<(Vec<api::TaskLogEntry>, u64), TaskError> {
        let tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let Some(state) = tasks.get(id) else {
            return Err(TaskError::NotFound);
        };
        if &state.owner != user_id {
            return Err(TaskError::NotFound);
        }
        let entries: Vec<api::TaskLogEntry> = state
            .logs
            .iter()
            .filter(|e| e.seq > after_seq)
            .take(limit as usize)
            .cloned()
            .collect();
        let next_seq = entries.last().map(|e| e.seq + 1).unwrap_or(state.next_seq);
        Ok((entries, next_seq))
    }

    /// Subscribe to `Event::Task*` events for `user_id`. Slow consumers
    /// drop oldest events (broadcast semantics) — the registry never
    /// applies back-pressure to task bodies.
    pub fn subscribe(&self, user_id: &UserId) -> broadcast::Receiver<api::Event> {
        let mut channels = self.inner.user_channels.lock().expect("channels poisoned");
        let sender = channels
            .entry(user_id.clone())
            .or_insert_with(|| broadcast::channel(self.inner.config.broadcast_capacity).0);
        sender.subscribe()
    }

    /// Append a log entry to `id`'s log ring from outside the task's
    /// body. Used by callers that produce events on behalf of a task
    /// they don't own the body of — for example, the `spawn_subagent`
    /// tool body (#112) appends a `ToolCall` entry to the *parent*
    /// task's log carrying the child task / conversation ids so the
    /// UI can render the spawn as a tool-call row that links to the
    /// child's panel.
    ///
    /// Cross-user safety: an attempt by a different user, or against
    /// an unknown id, is a silent no-op — same opacity rule as
    /// [`Self::get`] / [`Self::logs`], and consistent with
    /// [`TaskLogSink::append`]'s behaviour for an already-removed
    /// task. Callers should treat logging as best-effort and never
    /// observe a logging error.
    pub fn append_log(
        &self,
        user_id: &UserId,
        id: &api::TaskId,
        level: api::LogLevel,
        category: api::LogCategory,
        message: String,
        data: Option<serde_json::Value>,
    ) {
        let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let Some(state) = tasks.get_mut(id) else {
            return;
        };
        if &state.owner != user_id {
            return;
        }
        let entry = api::TaskLogEntry {
            seq: state.next_seq,
            timestamp: now_ms(),
            level,
            category,
            message,
            data,
        };
        state.next_seq += 1;
        if state.logs.len() == self.inner.config.log_ring_capacity {
            state.logs.pop_front();
        }
        state.logs.push_back(entry.clone());
        let owner = state.owner.clone();
        let task_id = id.clone();
        drop(tasks);
        self.inner.broadcast(
            &owner,
            api::Event::TaskLogAppended {
                id: task_id.0,
                entry,
            },
        );
    }

    /// Cold-restart sweep (#115): mark every persisted, non-terminal
    /// task row as `Failed` and surface it in the in-memory registry so
    /// `list`/`get`/`logs` see the leftovers from the previous daemon
    /// incarnation.
    ///
    /// Best-effort resume policy until #129 lands:
    ///
    /// - `TaskKind::Conversation` and `TaskKind::Subagent` — marked
    ///   `Failed { last_error: "daemon restarted mid-turn" }`. The
    ///   conversation history is intact in `conversations`/`messages`;
    ///   the user re-prompts to continue.
    /// - `TaskKind::Standalone` — marked `Failed { last_error:
    ///   "daemon restarted; resume not yet implemented" }`. #129
    ///   replaces this branch with a real resume from persisted turn
    ///   state.
    ///
    /// Persisted log replay is OUT of scope. The in-memory log for a
    /// resumed row starts fresh, prefixed with a single `Lifecycle`
    /// entry summarising the outcome.
    ///
    /// Returns the number of rows surfaced. Errors short-circuit so a
    /// transient DB failure doesn't leave the registry half-populated.
    pub async fn sweep_non_terminal_on_startup(&self) -> Result<u32, anyhow::Error> {
        let Some(store) = self.inner.store.as_ref() else {
            return Ok(0);
        };
        let mut rows = store.scan_non_terminal().await?;
        // Sort so rows without a parent come first: this guarantees a
        // parent is in the in-memory map before we try to register a
        // child under it, regardless of the order the DB returns them
        // in (sqlx makes no ordering guarantees). A child whose parent
        // is itself missing (e.g. parent finished but child still
        // pending) keeps its `parent` field but the parent's
        // `children` vector won't gain the entry — same policy as
        // in-memory dropped parents.
        rows.sort_by_key(|r| r.parent_task_id.is_some());
        let mut count = 0u32;
        for row in rows {
            // Parse the kind so we can branch on the resume policy.
            let kind: api::TaskKind = match serde_json::from_value(row.kind_json.clone()) {
                Ok(k) => k,
                Err(e) => {
                    error!(
                        error = %e,
                        task_id = %row.id,
                        "parse kind_json during cold-restart sweep; skipping",
                    );
                    continue;
                }
            };
            let last_error = match &kind {
                api::TaskKind::Conversation { .. } | api::TaskKind::Subagent { .. } => {
                    "daemon restarted mid-turn"
                }
                api::TaskKind::Standalone { .. } => {
                    // Until #129 lands, standalone tasks can't resume —
                    // we mark them Failed but emit a distinct error
                    // message so the UI can surface the "we lost it"
                    // case differently from a genuine error.
                    "daemon restarted; resume not yet implemented"
                }
            };
            let now = now_ms();
            // Persist the terminal transition first so a second crash
            // doesn't re-surface the row endlessly.
            //
            // The store's `update_task` reads `current_user_id()` from
            // the task-local. The sweep runs at daemon boot before any
            // request scope is installed, so we wrap the call in a
            // `with_user_id` set to the row's owner. This mirrors the
            // discipline used elsewhere in the application layer when a
            // system task touches user-scoped storage.
            let owner = UserId::new(row.user_id.clone());
            let store_for_call = Arc::clone(store);
            let row_id_for_call = row.id.clone();
            let progress_for_call = row.progress_hint.clone();
            if let Err(e) = desktop_assistant_core::ports::auth::with_user_id(owner, async move {
                store_for_call
                    .update_task(
                        &row_id_for_call,
                        BackgroundTaskStatus::Failed,
                        Some(last_error),
                        progress_for_call.as_deref(),
                        Some(now),
                    )
                    .await
            })
            .await
            {
                warn!(error = %e, task_id = %row.id, "sweep update_task failed; skipping in-memory surface");
                continue;
            }

            // Surface the row in the in-memory registry so `list`/`get`
            // see it. Marked `completed = true` so `wait` returns
            // immediately and `cancel` rejects with `AlreadyTerminal`.
            let owner = UserId::new(row.user_id.clone());
            let view = api::TaskView {
                id: api::TaskId(row.id.clone()),
                kind: kind.clone(),
                status: api::TaskStatus::Failed,
                started_at: row.started_at,
                ended_at: Some(now),
                last_error: Some(last_error.to_string()),
                parent: row.parent_task_id.clone().map(api::TaskId),
                children: Vec::new(),
                title: row.title.clone(),
                progress_hint: row.progress_hint.clone(),
            };
            {
                let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
                tasks.insert(
                    view.id.clone(),
                    TaskState {
                        owner: owner.clone(),
                        view: view.clone(),
                        // The token is already inert; cancelling a
                        // terminal task is a programmer error caught
                        // upstream by `AlreadyTerminal`.
                        token: CancellationToken::new(),
                        logs: VecDeque::with_capacity(2),
                        next_seq: 1,
                        completed: true,
                    },
                );
            }

            // Emit a single lifecycle log so the UI sees a "we lost it"
            // marker when it inspects the task. We append directly into
            // the just-inserted state rather than going through the
            // `TaskLogSink` because the latter has no context to know
            // we want a specific message rather than the generic
            // "task started" line.
            {
                let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
                if let Some(state) = tasks.get_mut(&view.id) {
                    let entry = api::TaskLogEntry {
                        seq: state.next_seq,
                        timestamp: now,
                        level: api::LogLevel::Warn,
                        category: api::LogCategory::Lifecycle,
                        message: last_error.to_string(),
                        data: None,
                    };
                    state.next_seq += 1;
                    state.logs.push_back(entry);
                }
            }

            // Re-link the child to its parent if the parent also
            // surfaced via this sweep. We only walk in the inserted
            // direction; orphaned children (parent terminal already)
            // keep their `parent` field but the parent's `children`
            // vector won't contain them — that matches the policy
            // applied to in-memory dropped parents.
            if let Some(parent_id) = &view.parent {
                let mut tasks = self.inner.tasks.lock().expect("tasks poisoned");
                if let Some(parent_state) = tasks.get_mut(parent_id) {
                    parent_state.view.children.push(view.id.clone());
                }
            }

            // Broadcast TaskStarted then TaskCompleted so a UI that
            // subscribes immediately after restart still observes the
            // lifecycle. This mirrors what a real spawn+finalize would
            // have emitted.
            self.inner
                .broadcast(&owner, api::Event::TaskStarted { task: view.clone() });
            self.inner.broadcast(
                &owner,
                api::Event::TaskCompleted {
                    id: view.id.0.clone(),
                    status: api::TaskStatus::Failed,
                    last_error: Some(last_error.to_string()),
                },
            );

            count = count.saturating_add(1);
        }
        Ok(count)
    }

    /// Resolve when `id`'s task reaches a terminal state. Used by the
    /// foreground send-message wrapper to keep the old "await until
    /// done" contract while still routing through the registry.
    ///
    /// Returns immediately if the task is already terminal or unknown
    /// (the latter cannot happen if the caller just spawned the id).
    pub async fn wait(&self, id: &api::TaskId) {
        // Fast-path: already terminal or unknown.
        {
            let tasks = self.inner.tasks.lock().expect("tasks poisoned");
            match tasks.get(id) {
                Some(state) if state.completed => return,
                None => return,
                _ => {}
            }
        }

        // Slow-path: install a per-task notify (or reuse a previously
        // installed one for concurrent waiters) and `enable()` the
        // `Notified` future BEFORE the second completion check.
        //
        // The enable-before-check order matters: `finalize` calls
        // `notify_waiters`, which only wakes futures already enrolled
        // in the wake list. If we enrolled after `finalize` had fired,
        // the wake would be lost and `wait` would hang forever.
        let notify = {
            let mut map = self
                .inner
                .completion_notify
                .lock()
                .expect("completion poisoned");
            Arc::clone(
                map.entry(id.clone())
                    .or_insert_with(|| Arc::new(tokio::sync::Notify::new())),
            )
        };

        let notified = notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();

        // Double-check completion AFTER enrolling so we close the race
        // window described above.
        {
            let tasks = self.inner.tasks.lock().expect("tasks poisoned");
            if tasks.get(id).map(|s| s.completed).unwrap_or(true) {
                return;
            }
        }

        notified.await;
    }
}

fn parent_task_id(kind: &api::TaskKind) -> Option<api::TaskId> {
    match kind {
        api::TaskKind::Subagent { parent_task_id, .. } => Some(parent_task_id.clone()),
        _ => None,
    }
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

struct TaskState {
    owner: UserId,
    view: api::TaskView,
    token: CancellationToken,
    logs: VecDeque<api::TaskLogEntry>,
    next_seq: u64,
    /// `true` once the task body has returned (terminal status set).
    completed: bool,
}

struct Inner {
    config: RegistryConfig,
    tasks: Mutex<HashMap<api::TaskId, TaskState>>,
    user_channels: Mutex<HashMap<UserId, broadcast::Sender<api::Event>>>,
    /// Per-task completion notifies, lazily created by `wait`.
    completion_notify: Mutex<HashMap<api::TaskId, Arc<tokio::sync::Notify>>>,
    /// Optional persistence layer (#115). When attached, spawned tasks
    /// are mirrored to the DB so a daemon restart can sweep abandoned
    /// rows. `None` keeps the registry purely in-memory (the default).
    store: Option<Arc<dyn BackgroundTaskStore>>,
}

impl Inner {
    /// Best-effort broadcast: a `SendError` (no live receivers) is normal
    /// and ignored — events that nobody is listening for are dropped.
    fn broadcast(&self, user_id: &UserId, event: api::Event) {
        let channels = self.user_channels.lock().expect("channels poisoned");
        if let Some(sender) = channels.get(user_id) {
            let _ = sender.send(event);
        }
    }

    /// Mirror a newly-spawned task to the persistence store. Logs and
    /// continues on failure — see the call site comment in `spawn` for
    /// the rationale.
    async fn persist_create(
        self: &Arc<Self>,
        task_id: &api::TaskId,
        user_id: &UserId,
        view: &api::TaskView,
    ) {
        let Some(store) = self.store.as_ref() else {
            return;
        };
        let kind_json = match serde_json::to_value(&view.kind) {
            Ok(v) => v,
            Err(e) => {
                // Serialization failure is a code bug — log loudly so
                // tests catch it but keep the task running.
                error!(
                    error = %e,
                    task_id = %task_id.0,
                    "serialize TaskKind for persistence",
                );
                return;
            }
        };
        let row = BackgroundTaskRow {
            id: task_id.0.clone(),
            user_id: user_id.as_str().to_string(),
            kind_json,
            status: api_status_to_db(view.status),
            parent_task_id: view.parent.as_ref().map(|p| p.0.clone()),
            title: view.title.clone(),
            last_error: view.last_error.clone(),
            progress_hint: view.progress_hint.clone(),
            started_at: view.started_at,
            ended_at: view.ended_at,
        };
        if let Err(e) = store.create_task(row).await {
            warn!(
                error = %e,
                task_id = %task_id.0,
                "persist background task on spawn",
            );
        }
    }

    /// Persist the terminal state of `task_id` to the store. Called from
    /// `finalize` once the in-memory transition is committed.
    async fn persist_update(
        self: &Arc<Self>,
        task_id: &api::TaskId,
        user_id: &UserId,
        status: BackgroundTaskStatus,
        last_error: Option<&str>,
        progress_hint: Option<&str>,
        ended_at: Option<i64>,
    ) {
        let Some(store) = self.store.as_ref() else {
            return;
        };
        // The store's `update_task` is scoped to `current_user_id()`.
        // The registry's task body runs without an installed user-id
        // task-local, so we wrap the call to ensure the WHERE clause
        // sees the right scope. This mirrors the discipline applied
        // by other application-layer call sites that bridge between
        // a registry's owned `UserId` and the storage layer's
        // task-local.
        let store = Arc::clone(store);
        let owner = user_id.clone();
        let task_id = task_id.0.clone();
        let last_error = last_error.map(String::from);
        let progress_hint = progress_hint.map(String::from);
        let result = desktop_assistant_core::ports::auth::with_user_id(owner, async move {
            store
                .update_task(
                    &task_id,
                    status,
                    last_error.as_deref(),
                    progress_hint.as_deref(),
                    ended_at,
                )
                .await
        })
        .await;
        if let Err(e) = result {
            warn!(error = %e, "persist background task on update");
        }
    }

    /// Transition `task_id` to a terminal state based on `result` and
    /// the cancellation-token state, broadcast `TaskCompleted`, and wake
    /// any waiters. Called from the `tokio::spawn` task wrapper.
    async fn finalize(
        self: &Arc<Self>,
        task_id: &api::TaskId,
        user_id: &UserId,
        result: anyhow::Result<()>,
    ) {
        let (status, last_error, progress_hint, ended_at, parent_id) = {
            let mut tasks = self.tasks.lock().expect("tasks poisoned");
            let Some(state) = tasks.get_mut(task_id) else {
                return;
            };
            let cancelled = state.token.is_cancelled();
            let (status, err): (api::TaskStatus, Option<String>) = match (result, cancelled) {
                // The token was tripped: regardless of whether the body
                // returned Ok or Err we count this as cancellation.
                (Ok(_), true) => (api::TaskStatus::Cancelled, None),
                (Err(e), true) => (api::TaskStatus::Cancelled, Some(e.to_string())),
                (Ok(_), false) => (api::TaskStatus::Completed, None),
                (Err(e), false) => (api::TaskStatus::Failed, Some(e.to_string())),
            };
            state.view.status = status;
            let ended_at = now_ms();
            state.view.ended_at = Some(ended_at);
            state.view.last_error = err.clone();
            state.completed = true;
            // Snapshot the fields needed for the post-broadcast persist
            // and eviction passes; the lock is released at scope end.
            (
                status,
                err,
                state.view.progress_hint.clone(),
                ended_at,
                state.view.parent.clone(),
            )
        };

        self.broadcast(
            user_id,
            api::Event::TaskCompleted {
                id: task_id.0.clone(),
                status,
                last_error: last_error.clone(),
            },
        );

        // Persist the terminal state. The in-memory broadcast happens
        // first because subscribers don't care about durability — they
        // care about the lifecycle event. Persistence happens after so
        // a slow DB doesn't gate the wake.
        self.persist_update(
            task_id,
            user_id,
            api_status_to_db(status),
            last_error.as_deref(),
            progress_hint.as_deref(),
            Some(ended_at),
        )
        .await;

        // Evict the terminal entry from the in-memory map (#158). The
        // broadcast has already fired and persistence has completed; a
        // missed broadcast subscriber can still recover the terminal
        // state from the persisted row. Holding terminal entries forever
        // would let the registry grow unbounded over the daemon's
        // lifetime.
        //
        // Done before notify_waiters so that any `wait`er that loops
        // back into `get`/`list` immediately observes the eviction.
        {
            let mut tasks = self.tasks.lock().expect("tasks poisoned");
            tasks.remove(task_id);
            if let Some(parent_id) = &parent_id
                && let Some(parent) = tasks.get_mut(parent_id)
            {
                parent.view.children.retain(|c| c != task_id);
            }
        }

        // Wake waiters.
        let waiter = {
            let mut map = self.completion_notify.lock().expect("completion poisoned");
            map.remove(task_id)
        };
        if let Some(notify) = waiter {
            notify.notify_waiters();
        }
    }
}

fn api_status_to_db(s: api::TaskStatus) -> BackgroundTaskStatus {
    match s {
        api::TaskStatus::Pending => BackgroundTaskStatus::Pending,
        api::TaskStatus::Running => BackgroundTaskStatus::Running,
        api::TaskStatus::Completed => BackgroundTaskStatus::Completed,
        api::TaskStatus::Failed => BackgroundTaskStatus::Failed,
        api::TaskStatus::Cancelled => BackgroundTaskStatus::Cancelled,
    }
}

/// Inverse of [`api_status_to_db`]. Currently used by sweep tests and
/// by callers that observe a `BackgroundTaskRow` (e.g. a future
/// "list across all daemons" command). Kept alongside the forward
/// mapping so the two stay symmetric.
#[allow(dead_code)]
fn db_status_to_api(s: BackgroundTaskStatus) -> api::TaskStatus {
    match s {
        BackgroundTaskStatus::Pending => api::TaskStatus::Pending,
        BackgroundTaskStatus::Running => api::TaskStatus::Running,
        BackgroundTaskStatus::Completed => api::TaskStatus::Completed,
        BackgroundTaskStatus::Failed => api::TaskStatus::Failed,
        BackgroundTaskStatus::Cancelled => api::TaskStatus::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finalize_sets_failed_on_error() {
        // Internal contract: an Err from the body and a non-cancelled
        // token yields TaskStatus::Failed with the error message in the
        // TaskCompleted event. The entry is evicted immediately, so we
        // observe status via subscribe rather than `get`.
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let mut events = registry.subscribe(&user);
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "failer".into(),
            |_ctx| async move { Err(anyhow::anyhow!("boom")) },
        );
        let want = id.0.clone();
        let (status, last_error) = loop {
            match tokio::time::timeout(std::time::Duration::from_secs(5), events.recv()).await {
                Ok(Ok(api::Event::TaskCompleted { id, status, last_error })) if id == want => {
                    break (status, last_error);
                }
                Ok(Ok(_)) => continue,
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(e)) => panic!("event channel closed: {e:?}"),
                Err(_) => panic!("timed out waiting for TaskCompleted"),
            }
        };
        assert_eq!(status, api::TaskStatus::Failed);
        assert_eq!(last_error.as_deref(), Some("boom"));
        assert!(registry.get(&user, &id).is_none(), "terminal task must be evicted");
    }

    #[tokio::test]
    async fn panicking_body_finalizes_as_failed_and_wakes_waiters() {
        // #171: a body that panics must still finalize (as Failed) rather
        // than leaving the row stuck `Running` and hanging every waiter
        // forever. The `tokio::time::timeout` around `wait()` is the
        // hang-detector: if this regresses, `wait()` never resolves and the
        // test fails fast instead of hanging the whole suite.
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let mut events = registry.subscribe(&user);
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "panicker".into(),
            |_ctx| async move {
                panic!("kaboom");
            },
        );

        // The bug: this `wait()` hung indefinitely. It must now resolve.
        tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&id))
            .await
            .expect("wait() on a panicked task must resolve, not hang");

        let want = id.0.clone();
        let (status, last_error) = loop {
            match tokio::time::timeout(std::time::Duration::from_secs(5), events.recv()).await {
                Ok(Ok(api::Event::TaskCompleted {
                    id,
                    status,
                    last_error,
                })) if id == want => break (status, last_error),
                Ok(Ok(_)) => continue,
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(e)) => panic!("event channel closed: {e:?}"),
                Err(_) => panic!("timed out waiting for TaskCompleted"),
            }
        };
        assert_eq!(status, api::TaskStatus::Failed);
        assert!(
            last_error
                .as_deref()
                .unwrap_or_default()
                .contains("panicked"),
            "last_error should note the panic, got {last_error:?}"
        );
        assert!(
            registry.get(&user, &id).is_none(),
            "panicked task must be evicted like any terminal task"
        );
    }

    #[tokio::test]
    async fn terminal_task_evicted_on_completion() {
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "ok".into(),
            |_ctx| async move { Ok(()) },
        );
        registry.wait(&id).await;
        assert!(registry.get(&user, &id).is_none());
        assert!(registry.list(&user, true, None).is_empty());
    }

    #[tokio::test]
    async fn terminal_task_evicted_on_cancellation() {
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "cancellable".into(),
            |ctx| async move {
                ctx.token.cancelled().await;
                Ok(())
            },
        );
        registry.cancel(&user, &id).expect("cancel");
        registry.wait(&id).await;
        assert!(registry.get(&user, &id).is_none());
    }

    #[tokio::test]
    async fn terminal_subagent_unlinked_from_parent_children() {
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let parent_release = Arc::new(tokio::sync::Notify::new());
        let parent_release_for_task = Arc::clone(&parent_release);
        let parent = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "parent".into(),
            move |_ctx| async move {
                parent_release_for_task.notified().await;
                Ok(())
            },
        );
        let child = registry.spawn(
            user.clone(),
            api::TaskKind::Subagent {
                parent_task_id: parent.clone(),
                conversation_id: "c-child".into(),
                name: "sub".into(),
            },
            "child".into(),
            |_ctx| async move { Ok(()) },
        );
        registry.wait(&child).await;

        let parent_view = registry.get(&user, &parent).expect("parent still running");
        assert!(
            !parent_view.children.contains(&child),
            "evicted child should be unlinked from parent.children, got {:?}",
            parent_view.children
        );

        parent_release.notify_one();
        registry.wait(&parent).await;
    }

    #[tokio::test]
    async fn wait_on_unknown_id_returns_immediately() {
        let registry = BackgroundTaskRegistry::new();
        // No-op fast-path: must not hang or panic.
        registry.wait(&api::TaskId("does-not-exist".into())).await;
    }

    #[tokio::test]
    async fn wait_does_not_lose_completion_notification_in_race_window() {
        // Regression: wait() must enroll its Notified future BEFORE
        // double-checking the completed flag. Otherwise finalize's
        // notify_waiters() can fire while wait is between the check
        // and the await, dropping the wake and hanging the test.
        // We sample many tight spawns to maximize the race surface.
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        for _ in 0..50 {
            let id = registry.spawn(
                user.clone(),
                api::TaskKind::Conversation {
                    conversation_id: "c".into(),
                },
                "race".into(),
                |_ctx| async move { Ok(()) },
            );
            // Cap with timeout so any regression manifests as a fail
            // instead of a hang.
            tokio::time::timeout(std::time::Duration::from_secs(5), registry.wait(&id))
                .await
                .expect("wait must not hang");
        }
    }

    #[tokio::test]
    async fn start_lifecycle_log_entry_is_broadcast_to_subscribers() {
        // The start-of-task Lifecycle marker is appended via TaskLogSink
        // and so is broadcast to subscribers. The completion is signaled
        // by the TaskCompleted event itself; a separate completion
        // lifecycle log would only live in state.logs which is wiped on
        // eviction, so we don't emit one.
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let mut events = registry.subscribe(&user);
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "lifecycle".into(),
            |_ctx| async move { Ok(()) },
        );
        let want = id.0.clone();
        let mut saw_start_marker = false;
        let mut saw_completed = false;
        while !saw_completed {
            match tokio::time::timeout(std::time::Duration::from_secs(5), events.recv()).await {
                Ok(Ok(api::Event::TaskLogAppended { id, entry }))
                    if id == want && entry.category == api::LogCategory::Lifecycle =>
                {
                    saw_start_marker = true;
                }
                Ok(Ok(api::Event::TaskCompleted { id, .. })) if id == want => {
                    saw_completed = true;
                }
                Ok(Ok(_)) => continue,
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                Ok(Err(e)) => panic!("event channel closed: {e:?}"),
                Err(_) => panic!("timed out waiting for events"),
            }
        }
        assert!(saw_start_marker, "expected start lifecycle marker");
    }
}
