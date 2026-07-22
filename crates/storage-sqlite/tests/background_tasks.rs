//! Contract tests for [`SqliteBackgroundTaskStore`] (issue #115).
#![cfg(feature = "sqlite")]

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore,
};
use desktop_assistant_storage_sqlite::{
    SqliteBackgroundTaskStore, UserId, create_memory_pool, with_user_id,
};

async fn store() -> SqliteBackgroundTaskStore {
    let pool = create_memory_pool().await.expect("pool");
    SqliteBackgroundTaskStore::new(pool)
}

fn row(id: &str, user: &str, status: BackgroundTaskStatus, started_at: i64) -> BackgroundTaskRow {
    BackgroundTaskRow {
        id: id.into(),
        user_id: user.into(),
        kind_json: serde_json::json!({"kind": "standalone"}),
        status,
        parent_task_id: None,
        title: format!("task {id}"),
        last_error: None,
        progress_hint: None,
        started_at,
        ended_at: None,
        owner_todo: String::new(),
        spawn_marker: None,
    }
}

#[tokio::test]
async fn create_get_update_roundtrips() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        s.create_task(row("t1", "alice", BackgroundTaskStatus::Running, 100))
            .await
            .expect("create");

        let got = s.get_task("t1").await.unwrap().expect("present");
        assert_eq!(got.status, BackgroundTaskStatus::Running);
        assert_eq!(got.kind_json, serde_json::json!({"kind": "standalone"}));

        s.update_task(
            "t1",
            BackgroundTaskStatus::Completed,
            None,
            Some("all done"),
            Some(200),
        )
        .await
        .expect("update");

        let got = s.get_task("t1").await.unwrap().unwrap();
        assert_eq!(got.status, BackgroundTaskStatus::Completed);
        assert_eq!(got.progress_hint.as_deref(), Some("all done"));
        assert_eq!(got.ended_at, Some(200));
    })
    .await;
}

#[tokio::test]
async fn duplicate_create_fails() {
    let s = store().await;
    s.create_task(row("t1", "u", BackgroundTaskStatus::Pending, 1))
        .await
        .unwrap();
    let err = s
        .create_task(row("t1", "u", BackgroundTaskStatus::Pending, 1))
        .await
        .unwrap_err();
    assert!(matches!(err, CoreError::Storage(_)));
}

#[tokio::test]
async fn get_is_scoped_per_user() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        s.create_task(row("t1", "alice", BackgroundTaskStatus::Running, 1))
            .await
            .unwrap();
    })
    .await;
    with_user_id(UserId::new("bob"), async {
        assert!(s.get_task("t1").await.unwrap().is_none());
    })
    .await;
}

#[tokio::test]
async fn list_for_user_orders_filters_and_limits() {
    let s = store().await;
    s.create_task(row("old", "alice", BackgroundTaskStatus::Running, 100))
        .await
        .unwrap();
    s.create_task(row("new", "alice", BackgroundTaskStatus::Running, 300))
        .await
        .unwrap();
    s.create_task(row("done", "alice", BackgroundTaskStatus::Completed, 200))
        .await
        .unwrap();
    // Another user's row must never appear in alice's list.
    s.create_task(row("bobs", "bob", BackgroundTaskStatus::Running, 999))
        .await
        .unwrap();

    // include_finished = false -> only pending/running, newest first.
    let active: Vec<String> = s
        .list_tasks_for_user("alice", false, None)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(active, vec!["new".to_string(), "old".to_string()]);

    // include_finished = true -> includes the completed one, ordered by
    // started_at DESC.
    let all: Vec<String> = s
        .list_tasks_for_user("alice", true, None)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(
        all,
        vec!["new".to_string(), "done".to_string(), "old".to_string()]
    );

    // limit caps the result.
    let capped = s.list_tasks_for_user("alice", true, Some(1)).await.unwrap();
    assert_eq!(capped.len(), 1);
    assert_eq!(capped[0].id, "new");
}

#[tokio::test]
async fn update_unknown_fails() {
    let s = store().await;
    let err = s
        .update_task(
            "ghost",
            BackgroundTaskStatus::Failed,
            Some("x"),
            None,
            Some(1),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, CoreError::Storage(_)));
}

#[tokio::test]
async fn scan_non_terminal_crosses_users_and_excludes_terminal() {
    let s = store().await;
    s.create_task(row("r", "alice", BackgroundTaskStatus::Running, 1))
        .await
        .unwrap();
    s.create_task(row("p", "bob", BackgroundTaskStatus::Pending, 2))
        .await
        .unwrap();
    s.create_task(row("c", "alice", BackgroundTaskStatus::Completed, 3))
        .await
        .unwrap();
    s.create_task(row("x", "bob", BackgroundTaskStatus::Cancelled, 4))
        .await
        .unwrap();

    let mut ids: Vec<String> = s
        .scan_non_terminal()
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["p".to_string(), "r".to_string()]);
}

#[tokio::test]
async fn owner_todo_and_marker_round_trip_sqlite() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        let mut r = row("t1", "alice", BackgroundTaskStatus::Running, 100);
        r.owner_todo = "1.1".into();
        r.spawn_marker = Some("marker-abc".into());
        s.create_task(r).await.expect("create");
        let got = s.get_task("t1").await.unwrap().expect("present");
        assert_eq!(got.owner_todo, "1.1");
        assert_eq!(got.spawn_marker.as_deref(), Some("marker-abc"));
    })
    .await;
}

#[tokio::test]
async fn legacy_row_defaults_to_empty_owner_todo_and_none_marker_sqlite() {
    // The `row` helper omits the new fields (pre-#287 caller shape); they
    // round-trip as '' / None.
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        s.create_task(row("t1", "alice", BackgroundTaskStatus::Running, 100))
            .await
            .expect("create");
        let got = s.get_task("t1").await.unwrap().unwrap();
        assert_eq!(got.owner_todo, "");
        assert_eq!(got.spawn_marker, None);
    })
    .await;
}

#[tokio::test]
async fn update_task_leaves_owner_todo_and_marker_unchanged_sqlite() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        let mut r = row("t1", "alice", BackgroundTaskStatus::Running, 100);
        r.owner_todo = "1.1".into();
        r.spawn_marker = Some("marker-abc".into());
        s.create_task(r).await.expect("create");
        s.update_task("t1", BackgroundTaskStatus::Completed, None, None, Some(200))
            .await
            .expect("update");
        let got = s.get_task("t1").await.unwrap().unwrap();
        assert_eq!(
            got.owner_todo, "1.1",
            "update_task must not touch owner_todo"
        );
        assert_eq!(
            got.spawn_marker.as_deref(),
            Some("marker-abc"),
            "update_task must not touch spawn_marker"
        );
        assert_eq!(got.status, BackgroundTaskStatus::Completed);
    })
    .await;
}
