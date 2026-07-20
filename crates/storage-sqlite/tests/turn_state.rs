//! Contract tests for [`SqliteTurnStateStore`] (issue #107).
#![cfg(feature = "sqlite")]

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{
    PendingClientToolCall, TurnRow, TurnStateJson, TurnStateStore, TurnStatus,
};
use desktop_assistant_storage_sqlite::{
    SqliteTurnStateStore, UserId, create_memory_pool, with_user_id,
};

async fn store() -> SqliteTurnStateStore {
    let pool = create_memory_pool().await.expect("pool");
    SqliteTurnStateStore::new(pool)
}

fn row(id: &str, user: &str, status: TurnStatus) -> TurnRow {
    TurnRow {
        id: id.into(),
        user_id: user.into(),
        conversation_id: "conv-1".into(),
        status,
        state: TurnStateJson::default(),
        last_error: None,
    }
}

#[tokio::test]
async fn create_get_update_roundtrips() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        s.create_turn(row("t1", "alice", TurnStatus::PendingLlm))
            .await
            .expect("create");

        let got = s.get_turn("t1").await.unwrap().expect("row present");
        assert_eq!(got.status, TurnStatus::PendingLlm);
        assert_eq!(got.conversation_id, "conv-1");

        let state = TurnStateJson {
            version: 1,
            pending_client_tool: Some(PendingClientToolCall {
                tool_call_id: "call-1".into(),
                tool_name: "fs_read".into(),
                arguments: serde_json::json!({"path": "/tmp/x"}),
            }),
        };
        s.update_turn("t1", TurnStatus::PendingClientTool, &state, Some("waiting"))
            .await
            .expect("update");

        let got = s.get_turn("t1").await.unwrap().unwrap();
        assert_eq!(got.status, TurnStatus::PendingClientTool);
        assert_eq!(got.last_error.as_deref(), Some("waiting"));
        assert_eq!(got.state.pending_client_tool.unwrap().tool_name, "fs_read");
    })
    .await;
}

#[tokio::test]
async fn duplicate_create_fails() {
    let s = store().await;
    s.create_turn(row("t1", "u", TurnStatus::PendingLlm))
        .await
        .unwrap();
    let err = s
        .create_turn(row("t1", "u", TurnStatus::PendingLlm))
        .await
        .unwrap_err();
    assert!(
        matches!(err, CoreError::Storage(_)),
        "a duplicate turn id must surface a Storage error, not silently overwrite"
    );
}

#[tokio::test]
async fn get_unknown_returns_none() {
    let s = store().await;
    assert!(s.get_turn("missing").await.unwrap().is_none());
}

#[tokio::test]
async fn get_and_update_are_scoped_per_user() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        s.create_turn(row("t1", "alice", TurnStatus::PendingLlm))
            .await
            .unwrap();
    })
    .await;

    with_user_id(UserId::new("bob"), async {
        // Bob cannot see or mutate Alice's turn.
        assert!(s.get_turn("t1").await.unwrap().is_none());
        assert!(matches!(
            s.update_turn("t1", TurnStatus::Failed, &TurnStateJson::default(), None)
                .await
                .unwrap_err(),
            CoreError::Storage(_)
        ));
    })
    .await;
}

#[tokio::test]
async fn update_unknown_fails() {
    let s = store().await;
    let err = s
        .update_turn(
            "ghost",
            TurnStatus::Complete,
            &TurnStateJson::default(),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, CoreError::Storage(_)));
}

#[tokio::test]
async fn scan_non_terminal_crosses_users_and_excludes_terminal() {
    let s = store().await;
    // Rows across two users, mixed statuses. The startup sweep bypasses the
    // per-user scope and returns only non-terminal rows.
    s.create_turn(row("done", "alice", TurnStatus::Complete))
        .await
        .unwrap();
    s.create_turn(row("failed", "alice", TurnStatus::Failed))
        .await
        .unwrap();
    s.create_turn(row("llm", "alice", TurnStatus::PendingLlm))
        .await
        .unwrap();
    s.create_turn(row("client", "bob", TurnStatus::PendingClientTool))
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
    assert_eq!(ids, vec!["client".to_string(), "llm".to_string()]);
}
