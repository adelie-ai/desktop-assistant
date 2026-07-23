//! Contract tests for [`SqliteConversationStore`] against an in-memory SQLite
//! database (no external service, fully deterministic).
//!
//! Each acceptance criterion is its own named test so a failing run names the
//! unmet behavior. Covers CRUD, the single-aggregate `list` projection with
//! `message_count`, message append/truncate on `update`, archive/unarchive,
//! reversible summaries (ON DELETE SET NULL), cross-`user_id` isolation, and
//! the JSON-column accessors (model selection / personality / tags).
#![cfg(feature = "sqlite")]

use desktop_assistant_core::domain::{Conversation, ConversationId, Message, Role};
use desktop_assistant_core::ports::inbound::ConversationModelSelection;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::prompts::{PersonalityLevel, PersonalityOverride};
use desktop_assistant_storage_sqlite::{
    SqliteConversationStore, UserId, create_memory_pool, with_user_id,
};

fn conv_with(id: &str, title: &str, updated_at: &str, n_messages: usize) -> Conversation {
    let mut conv = Conversation::new(id, title);
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = updated_at.to_string();
    for i in 0..n_messages {
        conv.messages
            .push(Message::new(Role::User, format!("msg {i}")));
    }
    conv
}

async fn store() -> SqliteConversationStore {
    let pool = create_memory_pool().await.expect("in-memory pool");
    SqliteConversationStore::new(pool)
}

#[tokio::test]
async fn create_then_get_roundtrips_all_fields() {
    let s = store().await;
    let mut conv = conv_with("c1", "Hello", "2026-01-02 03:04:05", 0);
    conv.context_summary = "rolling summary".into();
    conv.compacted_through = 3;
    conv.active_task = Some("do the thing".into());
    conv.tags = vec!["voice".into(), "urgent".into()];
    conv.messages.push(Message::new(Role::User, "hi there"));
    conv.messages
        .push(Message::new(Role::Assistant, "hello back"));

    s.create(conv.clone()).await.expect("create");
    let got = s.get(&ConversationId::from("c1")).await.expect("get");

    assert_eq!(got.title, "Hello");
    assert_eq!(got.created_at, "2026-01-01 00:00:00");
    assert_eq!(got.updated_at, "2026-01-02 03:04:05");
    assert_eq!(got.context_summary, "rolling summary");
    assert_eq!(got.compacted_through, 3);
    assert_eq!(got.active_task.as_deref(), Some("do the thing"));
    assert_eq!(got.tags, vec!["voice".to_string(), "urgent".to_string()]);
    assert_eq!(got.messages.len(), 2);
    assert_eq!(got.messages[0].content, "hi there");
    assert_eq!(got.messages[1].role, Role::Assistant);
    // The message's UUIDv7 identity is preserved across the round trip.
    assert_eq!(got.messages[0].id, conv.messages[0].id);
}

#[tokio::test]
async fn get_absent_conversation_is_not_found() {
    let s = store().await;
    let err = s.get(&ConversationId::from("nope")).await.unwrap_err();
    assert!(
        matches!(err, desktop_assistant_core::CoreError::ConversationNotFound(id) if id == "nope"),
        "absent conversation must surface ConversationNotFound"
    );
}

#[tokio::test]
async fn list_projects_message_count_and_orders_by_updated_desc() {
    let s = store().await;
    s.create(conv_with("older", "Older", "2026-01-01 00:00:00", 1))
        .await
        .unwrap();
    s.create(conv_with("newer", "Newer", "2026-03-01 00:00:00", 3))
        .await
        .unwrap();

    let list = s.list().await.expect("list");
    assert_eq!(list.len(), 2);
    // ORDER BY updated_at DESC.
    assert_eq!(list[0].id.as_str(), "newer");
    assert_eq!(list[1].id.as_str(), "older");
    // message_count comes from the aggregate LEFT JOIN, not a body fetch.
    assert_eq!(list[0].message_count, 3);
    assert_eq!(list[1].message_count, 1);
    assert!(!list[0].archived);
}

#[tokio::test]
async fn update_appends_and_truncates_messages() {
    let s = store().await;
    let mut conv = conv_with("c1", "T", "2026-01-01 00:00:00", 2);
    s.create(conv.clone()).await.unwrap();

    // Append one.
    conv.messages.push(Message::new(Role::Assistant, "third"));
    conv.updated_at = "2026-01-02 00:00:00".into();
    s.update(conv.clone()).await.unwrap();
    assert_eq!(
        s.get(&ConversationId::from("c1"))
            .await
            .unwrap()
            .messages
            .len(),
        3
    );

    // Truncate back to one.
    conv.messages.truncate(1);
    s.update(conv.clone()).await.unwrap();
    let got = s.get(&ConversationId::from("c1")).await.unwrap();
    assert_eq!(got.messages.len(), 1);
    assert_eq!(got.messages[0].content, "msg 0");
}

#[tokio::test]
async fn update_absent_conversation_fails() {
    let s = store().await;
    let conv = conv_with("ghost", "T", "2026-01-01 00:00:00", 0);
    let err = s.update(conv).await.unwrap_err();
    assert!(matches!(
        err,
        desktop_assistant_core::CoreError::ConversationNotFound(_)
    ));
}

#[tokio::test]
async fn delete_removes_conversation() {
    let s = store().await;
    s.create(conv_with("c1", "T", "2026-01-01 00:00:00", 2))
        .await
        .unwrap();
    s.delete(&ConversationId::from("c1")).await.expect("delete");
    assert!(s.get(&ConversationId::from("c1")).await.is_err());

    // Deleting again is a not-found error (nothing to remove).
    let err = s.delete(&ConversationId::from("c1")).await.unwrap_err();
    assert!(matches!(
        err,
        desktop_assistant_core::CoreError::ConversationNotFound(_)
    ));
}

#[tokio::test]
async fn archive_unarchive_lifecycle() {
    let s = store().await;
    s.create(conv_with("c1", "T", "2026-01-01 00:00:00", 0))
        .await
        .unwrap();

    s.archive(&ConversationId::from("c1"))
        .await
        .expect("archive");
    assert!(
        s.get(&ConversationId::from("c1"))
            .await
            .unwrap()
            .archived_at
            .is_some()
    );
    assert!(s.list().await.unwrap()[0].archived);

    // Re-archiving an already-archived conversation is a no-op success, not an
    // error (mirrors the Postgres adapter's opacity handling).
    s.archive(&ConversationId::from("c1"))
        .await
        .expect("re-archive is Ok");

    s.unarchive(&ConversationId::from("c1"))
        .await
        .expect("unarchive");
    assert!(
        s.get(&ConversationId::from("c1"))
            .await
            .unwrap()
            .archived_at
            .is_none()
    );
}

#[tokio::test]
async fn archive_absent_conversation_is_not_found() {
    let s = store().await;
    let err = s.archive(&ConversationId::from("ghost")).await.unwrap_err();
    assert!(matches!(
        err,
        desktop_assistant_core::CoreError::ConversationNotFound(_)
    ));
}

#[tokio::test]
async fn create_summary_stamps_range_and_expand_clears_via_set_null() {
    let s = store().await;
    let conv = conv_with("c1", "T", "2026-01-01 00:00:00", 3);
    s.create(conv).await.unwrap();

    let summary_id = s
        .create_summary(&ConversationId::from("c1"), "collapsed".into(), 0, 1)
        .await
        .expect("create_summary");

    let got = s.get(&ConversationId::from("c1")).await.unwrap();
    assert_eq!(
        got.messages[0].summary_id.as_deref(),
        Some(summary_id.as_str())
    );
    assert_eq!(
        got.messages[1].summary_id.as_deref(),
        Some(summary_id.as_str())
    );
    assert_eq!(
        got.messages[2].summary_id, None,
        "out-of-range message untouched"
    );
    assert_eq!(got.summaries.len(), 1);
    assert_eq!(got.summaries[0].summary, "collapsed");

    // Expanding deletes the summary; ON DELETE SET NULL clears the links.
    s.expand_summary(&summary_id).await.expect("expand");
    let got = s.get(&ConversationId::from("c1")).await.unwrap();
    assert!(got.messages.iter().all(|m| m.summary_id.is_none()));
    assert!(got.summaries.is_empty());
}

#[tokio::test]
async fn conversations_are_isolated_across_users() {
    let s = store().await;
    with_user_id(UserId::new("alice"), async {
        s.create(conv_with("c1", "Alice's", "2026-01-01 00:00:00", 1))
            .await
            .unwrap();
    })
    .await;

    with_user_id(UserId::new("bob"), async {
        // Cross-user reads behave like the row doesn't exist (no existence leak).
        assert!(s.get(&ConversationId::from("c1")).await.is_err());
        assert!(s.list().await.unwrap().is_empty());
        assert!(matches!(
            s.delete(&ConversationId::from("c1")).await.unwrap_err(),
            desktop_assistant_core::CoreError::ConversationNotFound(_)
        ));
    })
    .await;

    // Alice still sees her own conversation.
    with_user_id(UserId::new("alice"), async {
        assert_eq!(s.list().await.unwrap().len(), 1);
    })
    .await;
}

#[tokio::test]
async fn model_selection_set_get_and_clear() {
    let s = store().await;
    s.create(conv_with("c1", "T", "2026-01-01 00:00:00", 0))
        .await
        .unwrap();
    let id = ConversationId::from("c1");

    assert_eq!(
        s.get_conversation_model_selection(&id).await.unwrap(),
        None,
        "unset selection reads as None"
    );

    let sel = ConversationModelSelection {
        connection_id: "openai".into(),
        model_id: "gpt-x".into(),
        effort: None,
    };
    s.set_conversation_model_selection(&id, Some(&sel))
        .await
        .unwrap();
    assert_eq!(
        s.get_conversation_model_selection(&id).await.unwrap(),
        Some(sel)
    );

    s.set_conversation_model_selection(&id, None).await.unwrap();
    assert_eq!(s.get_conversation_model_selection(&id).await.unwrap(), None);
}

#[tokio::test]
async fn model_selection_on_absent_conversation_is_not_found() {
    let s = store().await;
    let err = s
        .get_conversation_model_selection(&ConversationId::from("ghost"))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        desktop_assistant_core::CoreError::ConversationNotFound(_)
    ));
}

#[tokio::test]
async fn personality_override_set_and_get() {
    let s = store().await;
    s.create(conv_with("c1", "T", "2026-01-01 00:00:00", 0))
        .await
        .unwrap();
    let id = ConversationId::from("c1");

    assert_eq!(s.get_conversation_personality(&id).await.unwrap(), None);

    let ovr = PersonalityOverride {
        humor: Some(PersonalityLevel::Never),
        ..Default::default()
    };
    s.set_conversation_personality(&id, Some(&ovr))
        .await
        .unwrap();
    assert_eq!(
        s.get_conversation_personality(&id).await.unwrap(),
        Some(ovr)
    );
}

// --- #570 Phase 1b: idempotency-key persistence + load-surfacing ------------

/// A user message stamped with an idempotency key persists it, and `get`
/// surfaces the key back onto the domain `Message` (mirrors the Postgres store).
#[tokio::test]
async fn user_message_idempotency_key_round_trips_through_get() {
    let s = store().await;
    let mut conv = conv_with("c1", "T", "2026-01-01 00:00:00", 0);
    let mut user_msg = Message::new(Role::User, "remember me");
    user_msg.idempotency_key = Some("idem-key-1".to_string());
    conv.messages.push(user_msg);

    s.create(conv).await.expect("create");
    let got = s.get(&ConversationId::from("c1")).await.expect("get");
    assert_eq!(got.messages.len(), 1);
    assert_eq!(
        got.messages[0].idempotency_key.as_deref(),
        Some("idem-key-1"),
        "the persisted user idempotency_key must round-trip through get"
    );
}

/// Assistant rows never carry an idempotency key; the user row that opened the
/// turn does.
#[tokio::test]
async fn assistant_message_idempotency_key_persists_null() {
    let s = store().await;
    let mut conv = conv_with("c1", "T", "2026-01-01 00:00:00", 0);
    let mut user_msg = Message::new(Role::User, "hello");
    user_msg.idempotency_key = Some("k-user".to_string());
    conv.messages.push(user_msg);
    conv.messages
        .push(Message::new(Role::Assistant, "hi back"));

    s.create(conv).await.expect("create");
    let got = s.get(&ConversationId::from("c1")).await.expect("get");
    assert_eq!(got.messages[0].idempotency_key.as_deref(), Some("k-user"));
    assert_eq!(
        got.messages[1].idempotency_key, None,
        "assistant rows never carry a client idempotency key"
    );
}

/// A keyless user message loads with `idempotency_key == None`.
#[tokio::test]
async fn user_message_without_key_loads_as_none() {
    let s = store().await;
    let mut conv = conv_with("c1", "T", "2026-01-01 00:00:00", 0);
    conv.messages
        .push(Message::new(Role::User, "no key here"));

    s.create(conv).await.expect("create");
    let got = s.get(&ConversationId::from("c1")).await.expect("get");
    assert_eq!(
        got.messages[0].idempotency_key, None,
        "a keyless user message loads with a None idempotency_key"
    );
}

#[tokio::test]
async fn get_conversation_tags_reads_stored_tags() {
    let s = store().await;
    let mut conv = conv_with("c1", "T", "2026-01-01 00:00:00", 0);
    conv.tags = vec!["voice".into()];
    s.create(conv).await.unwrap();
    assert_eq!(
        s.get_conversation_tags(&ConversationId::from("c1"))
            .await
            .unwrap(),
        vec!["voice".to_string()]
    );
    // Absent conversation fails closed to empty tags (no error).
    assert!(
        s.get_conversation_tags(&ConversationId::from("ghost"))
            .await
            .unwrap()
            .is_empty()
    );
}
