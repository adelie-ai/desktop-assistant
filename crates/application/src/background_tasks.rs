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
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

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
            }),
        }
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
                    logs: VecDeque::with_capacity(
                        self.inner.config.log_ring_capacity.min(64),
                    ),
                    // Seq numbers start at 1 so callers can pass
                    // `after_seq=0` to mean "I've seen nothing yet" —
                    // the filter in [`BackgroundTaskRegistry::logs`] is
                    // strict-greater-than.
                    next_seq: 1,
                    completed: false,
                },
            );
            if let Some(parent_id) = &parent
                && let Some(parent_state) = tasks.get_mut(parent_id) {
                    parent_state.view.children.push(task_id.clone());
                }
        }

        self.inner.broadcast(
            &user_id,
            api::Event::TaskStarted {
                task: view.clone(),
            },
        );

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
        tokio::spawn(async move {
            // Run the body. We always finalize, even on panic, via a
            // drop-guard so the registry never gets stuck with a row
            // in `Running` after the task disappears.
            let result = body(ctx_for_body).await;
            inner.finalize(&task_id_for_body, &user_id_for_body, result);
        });

        task_id
    }

    /// Request cancellation of `id` owned by `user_id`. Cooperative — the
    /// running future is responsible for noticing.
    pub fn cancel(&self, user_id: &UserId, id: &api::TaskId) -> Result<(), TaskError> {
        let tasks = self.inner.tasks.lock().expect("tasks poisoned");
        let Some(state) = tasks.get(id) else {
            return Err(TaskError::NotFound);
        };
        if &state.owner != user_id {
            // Existence-hiding: pretend it doesn't exist (#105 contract).
            return Err(TaskError::NotFound);
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
        let next_seq = entries
            .last()
            .map(|e| e.seq + 1)
            .unwrap_or(state.next_seq);
        Ok((entries, next_seq))
    }

    /// Subscribe to `Event::Task*` events for `user_id`. Slow consumers
    /// drop oldest events (broadcast semantics) — the registry never
    /// applies back-pressure to task bodies.
    pub fn subscribe(&self, user_id: &UserId) -> broadcast::Receiver<api::Event> {
        let mut channels = self.inner.user_channels.lock().expect("channels poisoned");
        let sender = channels.entry(user_id.clone()).or_insert_with(|| {
            broadcast::channel(self.inner.config.broadcast_capacity).0
        });
        sender.subscribe()
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

    /// Transition `task_id` to a terminal state based on `result` and
    /// the cancellation-token state, broadcast `TaskCompleted`, and wake
    /// any waiters. Called from the `tokio::spawn` task wrapper.
    fn finalize(
        self: &Arc<Self>,
        task_id: &api::TaskId,
        user_id: &UserId,
        result: anyhow::Result<()>,
    ) {
        let (status, last_error) = {
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
            state.view.ended_at = Some(now_ms());
            state.view.last_error = err.clone();
            state.completed = true;
            (status, err)
        };

        // Emit a lifecycle log line so a late subscriber that only reads
        // logs still sees the terminal marker.
        {
            let mut tasks = self.tasks.lock().expect("tasks poisoned");
            if let Some(state) = tasks.get_mut(task_id) {
                let entry = api::TaskLogEntry {
                    seq: state.next_seq,
                    timestamp: now_ms(),
                    level: match status {
                        api::TaskStatus::Failed => api::LogLevel::Error,
                        api::TaskStatus::Cancelled => api::LogLevel::Warn,
                        _ => api::LogLevel::Info,
                    },
                    category: api::LogCategory::Lifecycle,
                    message: match status {
                        api::TaskStatus::Completed => "task completed".into(),
                        api::TaskStatus::Cancelled => "task cancelled".into(),
                        api::TaskStatus::Failed => "task failed".into(),
                        _ => "task terminated".into(),
                    },
                    data: None,
                };
                state.next_seq += 1;
                if state.logs.len() == self.config.log_ring_capacity {
                    state.logs.pop_front();
                }
                state.logs.push_back(entry);
            }
        }

        self.broadcast(
            user_id,
            api::Event::TaskCompleted {
                id: task_id.0.clone(),
                status,
                last_error,
            },
        );

        // Wake waiters.
        let waiter = {
            let mut map = self
                .completion_notify
                .lock()
                .expect("completion poisoned");
            map.remove(task_id)
        };
        if let Some(notify) = waiter {
            notify.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finalize_sets_failed_on_error() {
        // Internal contract: an Err from the body and a non-cancelled
        // token yields TaskStatus::Failed with the error message.
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "failer".into(),
            |_ctx| async move { Err(anyhow::anyhow!("boom")) },
        );
        registry.wait(&id).await;
        let view = registry.get(&user, &id).expect("present");
        assert_eq!(view.status, api::TaskStatus::Failed);
        assert_eq!(view.last_error.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn wait_on_unknown_id_returns_immediately() {
        let registry = BackgroundTaskRegistry::new();
        // No-op fast-path: must not hang or panic.
        registry
            .wait(&api::TaskId("does-not-exist".into()))
            .await;
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
    async fn lifecycle_log_entries_emitted_on_start_and_completion() {
        let registry = BackgroundTaskRegistry::new();
        let user = UserId::new("alice");
        let id = registry.spawn(
            user.clone(),
            api::TaskKind::Conversation {
                conversation_id: "c".into(),
            },
            "lifecycle".into(),
            |_ctx| async move { Ok(()) },
        );
        registry.wait(&id).await;
        let (entries, _) = registry.logs(&user, &id, 0, 100).unwrap();
        let categories: Vec<_> = entries.iter().map(|e| e.category).collect();
        assert!(categories.contains(&api::LogCategory::Lifecycle));
        // Both start and completion lifecycle markers should be present.
        let count = categories
            .iter()
            .filter(|c| **c == api::LogCategory::Lifecycle)
            .count();
        assert_eq!(count, 2, "expected start + completion lifecycle markers");
    }
}
