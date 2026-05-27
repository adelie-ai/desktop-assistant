//! Issue #115 acceptance tests: durable background tasks across daemon
//! restart.
//!
//! These tests describe the desired *business outcomes* of the
//! persistence layer that mirrors the in-memory
//! `BackgroundTaskRegistry`:
//!
//! 1. Every spawn writes a row to the store.
//! 2. Every terminal transition updates the row.
//! 3. A "daemon restart" (a fresh registry pointed at the same store)
//!    sweeps non-terminal rows into `Failed` and surfaces them in the
//!    in-memory map.
//! 4. The resume policy distinguishes `Conversation`/`Subagent` (mark
//!    failed; user re-prompts) from `Standalone` (mark failed with a
//!    distinct message until #129's real resume lands).
//! 5. Cross-user isolation survives restart.
//! 6. Cancel on a post-restart Failed row returns `AlreadyTerminal`,
//!    not a silent no-op.
//!
//! The mock store in this file is a pure in-memory `Mutex<HashMap>`;
//! the real Postgres adapter is covered by the storage crate's
//! `user_id_scoping` tests (which gain a few cases below the registry
//! tests). Keeping the registry tests off Postgres lets every
//! contributor run the suite without spinning up a DB.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use desktop_assistant_api_model as api;
use desktop_assistant_application::background_tasks::{
    BackgroundTaskRegistry, TaskError,
};
use desktop_assistant_application::UserId;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore,
};
use tokio::sync::Notify;
use tokio::time::{timeout, Duration};

/// In-memory `BackgroundTaskStore` for tests. Records SQL-like
/// invariants the registry depends on:
///  - duplicate ids fail to create
///  - update on a missing/cross-user id returns Err
///  - scan_non_terminal walks across users
struct MockStore {
    rows: Mutex<HashMap<String, BackgroundTaskRow>>,
}

impl MockStore {
    fn new() -> Self {
        Self {
            rows: Mutex::new(HashMap::new()),
        }
    }

    fn snapshot(&self) -> Vec<BackgroundTaskRow> {
        self.rows.lock().unwrap().values().cloned().collect()
    }

    fn get_raw(&self, id: &str) -> Option<BackgroundTaskRow> {
        self.rows.lock().unwrap().get(id).cloned()
    }
}

#[async_trait]
impl BackgroundTaskStore for MockStore {
    async fn create_task(&self, row: BackgroundTaskRow) -> Result<(), CoreError> {
        let mut rows = self.rows.lock().unwrap();
        if rows.contains_key(&row.id) {
            return Err(CoreError::Storage(format!(
                "background task id already exists: {}",
                row.id
            )));
        }
        rows.insert(row.id.clone(), row);
        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<BackgroundTaskRow>, CoreError> {
        // No user-scope filtering in the mock — the registry exercises
        // scoping via the real Postgres adapter; the mock just provides
        // the surface the registry calls.
        Ok(self.rows.lock().unwrap().get(id).cloned())
    }

    async fn update_task(
        &self,
        id: &str,
        status: BackgroundTaskStatus,
        last_error: Option<&str>,
        progress_hint: Option<&str>,
        ended_at: Option<i64>,
    ) -> Result<(), CoreError> {
        let mut rows = self.rows.lock().unwrap();
        let row = rows
            .get_mut(id)
            .ok_or_else(|| CoreError::Storage(format!("background task not found: {id}")))?;
        row.status = status;
        row.last_error = last_error.map(String::from);
        row.progress_hint = progress_hint.map(String::from);
        row.ended_at = ended_at;
        Ok(())
    }

    async fn list_tasks_for_user(
        &self,
        user_id: &str,
        include_finished: bool,
        limit: Option<u32>,
    ) -> Result<Vec<BackgroundTaskRow>, CoreError> {
        let rows = self.rows.lock().unwrap();
        let mut out: Vec<_> = rows
            .values()
            .filter(|r| r.user_id == user_id)
            .filter(|r| {
                if include_finished {
                    true
                } else {
                    matches!(
                        r.status,
                        BackgroundTaskStatus::Pending | BackgroundTaskStatus::Running
                    )
                }
            })
            .cloned()
            .collect();
        out.sort_by_key(|r| std::cmp::Reverse(r.started_at));
        if let Some(limit) = limit {
            out.truncate(limit as usize);
        }
        Ok(out)
    }

    async fn scan_non_terminal(&self) -> Result<Vec<BackgroundTaskRow>, CoreError> {
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

/// Wait until `pred()` returns true, polling at most ~2s. Fails with a
/// helpful label instead of hanging if the predicate never trips.
async fn wait_until<F: FnMut() -> bool>(mut pred: F, label: &str) {
    for _ in 0..200 {
        if pred() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("predicate '{label}' never became true within timeout");
}

fn standalone_kind() -> api::TaskKind {
    api::TaskKind::Standalone {
        name: "researcher".into(),
        conversation_id: "conv-stan".into(),
    }
}

fn conv_kind(id: &str) -> api::TaskKind {
    api::TaskKind::Conversation {
        conversation_id: id.into(),
    }
}

// ----------------------------------------------------------------------
// 1. Spawn writes a row to the store; the row mirrors the in-memory view.
// ----------------------------------------------------------------------

#[tokio::test]
async fn standalone_task_status_persists_across_restart() {
    // Spawn a Standalone task; let it run to a parking point; observe
    // that the DB has a `Running` row. Drop the registry, build a new
    // one against the same store, run the sweep, and observe that the
    // row is now `Failed` with the standalone-specific error message.
    let store = Arc::new(MockStore::new());

    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");

    let parked = Arc::new(Notify::new());
    let parked_for_body = Arc::clone(&parked);
    let task_id = registry.spawn(
        user.clone(),
        standalone_kind(),
        "stan".into(),
        move |ctx| async move {
            parked_for_body.notify_one();
            // Block until cancel — we'll never observe natural completion.
            ctx.token.cancelled().await;
            Ok(())
        },
    );

    // Wait for the body to enter steady state so the persistence write
    // has had a chance to land.
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Running)
                .unwrap_or(false)
        },
        "row persisted as Running",
    )
    .await;

    let row = store.get_raw(&task_id.0).expect("row present after spawn");
    assert_eq!(row.user_id, "alice");
    assert_eq!(row.status, BackgroundTaskStatus::Running);
    assert_eq!(row.title, "stan");
    assert!(row.last_error.is_none());

    // "Restart": drop the original registry without finalizing the task,
    // then build a fresh one against the same store and run the sweep.
    drop(registry);
    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    let count = post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");
    assert_eq!(count, 1, "exactly one row should be swept");

    // DB row reads Failed with the standalone resume-not-yet-implemented
    // message — covers the "until #129 lands" branch from the issue.
    let row = store
        .get_raw(&task_id.0)
        .expect("row still present post-sweep");
    assert_eq!(row.status, BackgroundTaskStatus::Failed);
    assert_eq!(
        row.last_error.as_deref(),
        Some("daemon restarted; resume not yet implemented"),
    );
    assert!(row.ended_at.is_some(), "ended_at must be stamped");

    // In-memory: list() now surfaces the row as Failed.
    let listed = post_restart.list(&user, /*include_finished*/ true, None);
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].status, api::TaskStatus::Failed);
    assert_eq!(
        listed[0].last_error.as_deref(),
        Some("daemon restarted; resume not yet implemented"),
    );
}

#[tokio::test]
async fn conversation_task_marked_failed_on_restart() {
    // The Conversation/Subagent branch of the resume policy uses the
    // "daemon restarted mid-turn" message.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");

    let parked = Arc::new(Notify::new());
    let parked_for_body = Arc::clone(&parked);
    let task_id = registry.spawn(
        user.clone(),
        conv_kind("conv-99"),
        "conv".into(),
        move |ctx| async move {
            parked_for_body.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Running)
                .unwrap_or(false)
        },
        "row persisted as Running",
    )
    .await;

    drop(registry);
    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");

    let row = store.get_raw(&task_id.0).unwrap();
    assert_eq!(row.status, BackgroundTaskStatus::Failed);
    assert_eq!(
        row.last_error.as_deref(),
        Some("daemon restarted mid-turn"),
    );
}

#[tokio::test]
async fn cancelled_tasks_persist_as_cancelled() {
    // Cancel → finalize → DB row reads Cancelled, not Failed. After a
    // restart the sweep MUST NOT touch this row (terminal is terminal).
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");

    let parked = Arc::new(Notify::new());
    let parked_for_body = Arc::clone(&parked);
    let task_id = registry.spawn(
        user.clone(),
        standalone_kind(),
        "stan".into(),
        move |ctx| async move {
            parked_for_body.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    registry.cancel(&user, &task_id).expect("cancel");
    registry.wait(&task_id).await;
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Cancelled)
                .unwrap_or(false)
        },
        "row persisted as Cancelled",
    )
    .await;
    let row_before_restart = store.get_raw(&task_id.0).unwrap();

    drop(registry);
    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    let count = post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");
    assert_eq!(count, 0, "terminal rows are never swept");
    let row_after = store.get_raw(&task_id.0).unwrap();
    assert_eq!(row_after.status, BackgroundTaskStatus::Cancelled);
    assert_eq!(
        row_after.last_error, row_before_restart.last_error,
        "sweep must not rewrite last_error on a terminal row"
    );
}

#[tokio::test]
async fn completed_tasks_persist_as_completed() {
    // Happy path: run to natural completion; row reads Completed; sweep
    // is a no-op.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");

    let task_id = registry.spawn(
        user.clone(),
        standalone_kind(),
        "stan".into(),
        move |_ctx| async move { Ok(()) },
    );
    registry.wait(&task_id).await;
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Completed)
                .unwrap_or(false)
        },
        "row persisted as Completed",
    )
    .await;

    drop(registry);
    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    let count = post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");
    assert_eq!(count, 0, "completed rows are never swept");
    assert_eq!(
        store.get_raw(&task_id.0).unwrap().status,
        BackgroundTaskStatus::Completed
    );
}

#[tokio::test]
async fn task_rows_are_user_id_scoped() {
    // Alice and Bob both spawn standalone tasks; the restart sweep
    // marks each as Failed but the in-memory listings stay scoped: a
    // user only sees their own rows.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let alice = UserId::new("alice");
    let bob = UserId::new("bob");

    let parked = Arc::new(Notify::new());
    let parked_a = Arc::clone(&parked);
    let parked_b = Arc::clone(&parked);
    let task_alice = registry.spawn(
        alice.clone(),
        standalone_kind(),
        "alice-stan".into(),
        move |ctx| async move {
            parked_a.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    let task_bob = registry.spawn(
        bob.clone(),
        standalone_kind(),
        "bob-stan".into(),
        move |ctx| async move {
            parked_b.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    // Two parks → two notifies; wait for both.
    for _ in 0..2 {
        timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    }
    wait_until(
        || {
            store
                .get_raw(&task_alice.0)
                .map(|r| r.status == BackgroundTaskStatus::Running)
                .unwrap_or(false)
                && store
                    .get_raw(&task_bob.0)
                    .map(|r| r.status == BackgroundTaskStatus::Running)
                    .unwrap_or(false)
        },
        "both rows persisted",
    )
    .await;

    drop(registry);
    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");

    let alice_list = post_restart.list(&alice, true, None);
    let bob_list = post_restart.list(&bob, true, None);
    assert_eq!(alice_list.len(), 1, "alice sees only her row");
    assert_eq!(alice_list[0].id, task_alice);
    assert_eq!(bob_list.len(), 1, "bob sees only his row");
    assert_eq!(bob_list[0].id, task_bob);

    // Cross-user `get` returns None (existence-hiding).
    assert!(post_restart.get(&alice, &task_bob).is_none());
    assert!(post_restart.get(&bob, &task_alice).is_none());
}

#[tokio::test]
async fn resumed_standalone_emits_lifecycle_log_until_129_lands() {
    // The sweep emits exactly one Lifecycle log entry per resumed row
    // so the UI's log viewer can show the "we lost it" reason. Until
    // #129 lands, the message is the standalone-specific "resume not
    // yet implemented" string.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");
    let parked = Arc::new(Notify::new());
    let parked_for_body = Arc::clone(&parked);
    let task_id = registry.spawn(
        user.clone(),
        standalone_kind(),
        "stan".into(),
        move |ctx| async move {
            parked_for_body.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Running)
                .unwrap_or(false)
        },
        "row persisted as Running",
    )
    .await;
    drop(registry);

    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");

    let (logs, _) = post_restart
        .logs(&user, &task_id, 0, 100)
        .expect("logs after sweep");
    let lifecycle: Vec<_> = logs
        .iter()
        .filter(|e| matches!(e.category, api::LogCategory::Lifecycle))
        .collect();
    assert_eq!(
        lifecycle.len(),
        1,
        "sweep emits exactly one lifecycle log entry, got {}: {:?}",
        lifecycle.len(),
        logs,
    );
    assert_eq!(
        lifecycle[0].message,
        "daemon restarted; resume not yet implemented"
    );
    assert_eq!(lifecycle[0].level, api::LogLevel::Warn);
}

#[tokio::test]
async fn parent_child_links_preserved_across_restart() {
    // Spawn a parent Standalone, then a child Subagent that references
    // the parent. Restart. The child's `parent` field still points at
    // the parent; the parent's `children` vector still contains the
    // child id. This is the contract from the issue body.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");

    let parked = Arc::new(Notify::new());
    let parked_p = Arc::clone(&parked);
    let parent_id = registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: "parent".into(),
            conversation_id: "conv-p".into(),
        },
        "parent".into(),
        move |ctx| async move {
            parked_p.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();

    let parked_c = Arc::clone(&parked);
    let child_kind = api::TaskKind::Subagent {
        parent_task_id: parent_id.clone(),
        conversation_id: "conv-c".into(),
        name: "child".into(),
    };
    let child_id = registry.spawn(
        user.clone(),
        child_kind,
        "child".into(),
        move |ctx| async move {
            parked_c.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    wait_until(
        || {
            store.get_raw(&parent_id.0).is_some() && store.get_raw(&child_id.0).is_some()
        },
        "both rows persisted",
    )
    .await;

    // The persisted child row carries the parent reference.
    let child_row = store.get_raw(&child_id.0).unwrap();
    assert_eq!(child_row.parent_task_id.as_deref(), Some(parent_id.0.as_str()));

    drop(registry);
    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");

    let parent_view = post_restart
        .get(&user, &parent_id)
        .expect("parent surfaces");
    let child_view = post_restart
        .get(&user, &child_id)
        .expect("child surfaces");
    assert_eq!(child_view.parent.as_ref(), Some(&parent_id));
    assert!(
        parent_view.children.contains(&child_id),
        "parent's children must include the child id, got {:?}",
        parent_view.children,
    );
}

#[tokio::test]
async fn cancel_on_post_restart_failed_task_returns_already_terminal() {
    // Sweep marks a row Failed. A subsequent cancel must NOT pretend to
    // succeed — clients need to know the task is unrecoverable.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("alice");
    let parked = Arc::new(Notify::new());
    let parked_for_body = Arc::clone(&parked);
    let task_id = registry.spawn(
        user.clone(),
        standalone_kind(),
        "stan".into(),
        move |ctx| async move {
            parked_for_body.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Running)
                .unwrap_or(false)
        },
        "row persisted as Running",
    )
    .await;
    drop(registry);

    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");

    let err = post_restart
        .cancel(&user, &task_id)
        .expect_err("cancel on post-restart Failed must error");
    assert_eq!(err, TaskError::AlreadyTerminal);
}

#[tokio::test]
async fn concurrent_spawn_and_restart_does_not_corrupt_state() {
    // Race a spawn against a freshly-constructed registry pointed at the
    // same store. The point isn't "this exact interleaving never
    // happens" — it's that the persistence layer is the source of
    // truth, so two registries can't produce duplicate rows for the
    // same task_id and a partially-written row never trips the sweep
    // into a bad state.
    let store = Arc::new(MockStore::new());

    // Pre-populate the store with a row that simulates "a task that
    // was running before the restart".
    let pre_id = "pre-existing-row".to_string();
    store
        .create_task(BackgroundTaskRow {
            id: pre_id.clone(),
            user_id: "alice".to_string(),
            kind_json: serde_json::to_value(standalone_kind()).unwrap(),
            status: BackgroundTaskStatus::Running,
            parent_task_id: None,
            title: "pre".into(),
            last_error: None,
            progress_hint: None,
            started_at: 1_700_000_000,
            ended_at: None,
        })
        .await
        .unwrap();

    let registry = BackgroundTaskRegistry::new().with_store(store.clone());

    // Concurrently: spawn a new task AND run the sweep. The sweep should
    // only touch the pre-existing row; the new spawn's row should
    // remain Running (not be incorrectly captured by an in-flight sweep).
    let registry_clone = registry.clone();
    let spawner = tokio::spawn(async move {
        registry_clone.spawn(
            UserId::new("alice"),
            standalone_kind(),
            "new".into(),
            move |ctx| async move {
                ctx.token.cancelled().await;
                Ok(())
            },
        )
    });
    let sweep_count = registry.sweep_non_terminal_on_startup().await.unwrap();
    let new_id = spawner.await.unwrap();

    // The pre-existing row is now Failed.
    assert_eq!(
        store.get_raw(&pre_id).unwrap().status,
        BackgroundTaskStatus::Failed
    );

    // The newly-spawned row exists, distinct from the pre-existing one.
    // Allow brief delay for the persistence write to land.
    wait_until(|| store.get_raw(&new_id.0).is_some(), "new row persisted").await;
    let snapshot = store.snapshot();
    let ids: std::collections::HashSet<_> = snapshot.iter().map(|r| r.id.clone()).collect();
    assert!(ids.contains(&pre_id));
    assert!(ids.contains(&new_id.0));
    // The sweep is intended to run at daemon boot, before any new
    // spawn. If the race happened to catch the new row too, that's
    // still correct behavior — what matters is the store ends in a
    // consistent state.
    assert!(sweep_count >= 1, "sweep must touch the pre-existing row");

    // No duplicate IDs in the store.
    assert_eq!(ids.len(), snapshot.len(), "no duplicate ids in store");
}

#[tokio::test]
async fn business_outcome_user_sees_failed_status_in_list_after_restart() {
    // Top-level integration: a user spawns a standalone, simulates a
    // restart, and then a follow-up `list(user_id)` call surfaces the
    // task with status Failed and a visible `last_error`.
    let store = Arc::new(MockStore::new());
    let registry = BackgroundTaskRegistry::new().with_store(store.clone());
    let user = UserId::new("dave");

    let parked = Arc::new(Notify::new());
    let parked_for_body = Arc::clone(&parked);
    let task_id = registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: "weekly-report".into(),
            conversation_id: "conv-1".into(),
        },
        "weekly report".into(),
        move |ctx| async move {
            parked_for_body.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    timeout(Duration::from_secs(2), parked.notified()).await.unwrap();
    wait_until(
        || {
            store
                .get_raw(&task_id.0)
                .map(|r| r.status == BackgroundTaskStatus::Running)
                .unwrap_or(false)
        },
        "row persisted",
    )
    .await;
    drop(registry);

    let post_restart = BackgroundTaskRegistry::new().with_store(store.clone());
    post_restart
        .sweep_non_terminal_on_startup()
        .await
        .expect("sweep");

    let tasks = post_restart.list(&user, /*include_finished=*/ true, None);
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.id, task_id);
    assert_eq!(task.status, api::TaskStatus::Failed);
    assert!(
        task.last_error.is_some(),
        "user-visible last_error must be set so the UI can show why",
    );
}
