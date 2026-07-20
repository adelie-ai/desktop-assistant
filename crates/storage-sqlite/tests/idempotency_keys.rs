//! Contract tests for [`SqliteIdempotencyKeyStore`] (issue #204). Rows are
//! keyed by `(user_id, conversation_id, idempotency_key)` and cascade-delete
//! with the parent conversation.
#![cfg(feature = "sqlite")]

use desktop_assistant_core::domain::{Conversation, ConversationId, Message, Role};
use desktop_assistant_core::ports::store::{ConversationStore, IdempotencyKeyStore};
use desktop_assistant_storage_sqlite::{
    SqliteConversationStore, SqliteIdempotencyKeyStore, SqlitePool, UserId, create_memory_pool,
    with_user_id,
};

async fn pool() -> SqlitePool {
    create_memory_pool().await.expect("pool")
}

fn make_conversation(id: &str) -> Conversation {
    let mut conv = Conversation::new(id, "idempotency test");
    conv.created_at = "2026-06-05 00:00:00".into();
    conv.updated_at = "2026-06-05 00:00:00".into();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

#[tokio::test]
async fn record_then_lookup_roundtrips() {
    let pool = pool().await;
    let convs = SqliteConversationStore::new(pool.clone());
    let store = SqliteIdempotencyKeyStore::new(pool);
    with_user_id(UserId::new("alice"), async {
        convs.create(make_conversation("c1")).await.unwrap();

        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap(),
            None,
            "an absent key must miss"
        );

        store
            .record_response("c1", "k1", "req-1", "the answer")
            .await
            .unwrap();

        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap().as_deref(),
            Some("the answer")
        );
    })
    .await;
}

#[tokio::test]
async fn record_is_idempotent_upsert() {
    let pool = pool().await;
    let convs = SqliteConversationStore::new(pool.clone());
    let store = SqliteIdempotencyKeyStore::new(pool);
    with_user_id(UserId::new("alice"), async {
        convs.create(make_conversation("c1")).await.unwrap();
        store
            .record_response("c1", "k1", "req-1", "first")
            .await
            .unwrap();
        // Re-recording the same key overwrites instead of erroring/duplicating,
        // so a turn that raced past the dedup check converges to one row.
        store
            .record_response("c1", "k1", "req-2", "second")
            .await
            .unwrap();
        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap().as_deref(),
            Some("second")
        );
    })
    .await;
}

#[tokio::test]
async fn lookup_is_scoped_per_user() {
    let pool = pool().await;
    let convs = SqliteConversationStore::new(pool.clone());
    let store = SqliteIdempotencyKeyStore::new(pool);

    with_user_id(UserId::new("alice"), async {
        convs.create(make_conversation("c1")).await.unwrap();
        store
            .record_response("c1", "k1", "r", "alice-secret")
            .await
            .unwrap();
    })
    .await;

    with_user_id(UserId::new("bob"), async {
        // Bob owns his own conversation with the same id + key; he must not see
        // alice's stored reply.
        convs.create(make_conversation("c1")).await.unwrap();
        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap(),
            None,
            "another user's identical key must be invisible"
        );
    })
    .await;
}

#[tokio::test]
async fn rows_cascade_delete_with_conversation() {
    let pool = pool().await;
    let convs = SqliteConversationStore::new(pool.clone());
    let store = SqliteIdempotencyKeyStore::new(pool);
    with_user_id(UserId::new("alice"), async {
        convs.create(make_conversation("c1")).await.unwrap();
        store
            .record_response("c1", "k1", "r", "answer")
            .await
            .unwrap();

        convs.delete(&ConversationId::from("c1")).await.unwrap();

        assert_eq!(
            store.lookup_completed("c1", "k1").await.unwrap(),
            None,
            "the idempotency row is cascade-deleted with its conversation"
        );
    })
    .await;
}
