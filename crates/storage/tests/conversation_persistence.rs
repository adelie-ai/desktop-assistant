//! Persistence-model tests for `ConversationStore::update` (review finding
//! DS-5): updates must diff-and-write instead of delete-all + re-insert.
//!
//! The observable contract under test:
//! - appending one message to an N-message conversation INSERTs exactly one
//!   row (proven via row-id stability: regenerated rows would get fresh
//!   `now_v7()` ids);
//! - unchanged messages keep their row ids across updates;
//! - metadata-only changes (summary_id) update rows in place;
//! - truncation deletes only the removed tail;
//! - user scoping is not weakened.
//!
//! ## Running locally
//!
//! Same harness as `user_id_scoping.rs`:
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test conversation_persistence
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, Message, Role, ToolCall};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{PgConversationStore, UserId, run_migrations, with_user_id};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// RAII fixture: private schema, pool pinned to it, migrations applied.
struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("ds5_{}", Uuid::now_v7().simple());

        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to TEST_DATABASE_URL");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA \"{schema}\"")))
            .execute(&admin)
            .await
            .expect("create test schema");
        admin.close().await;

        let schema_for_hook = Arc::new(schema.clone());
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .after_connect(move |conn, _| {
                let schema = Arc::clone(&schema_for_hook);
                Box::pin(async move {
                    let sql = format!("SET search_path TO \"{schema}\", public");
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&url)
            .await
            .expect("connect per-test pool");

        run_migrations(&pool)
            .await
            .expect("run_migrations succeeds");

        Some(Self {
            pool,
            schema,
            admin_url: url,
        })
    }

    async fn cleanup(self) {
        self.pool.close().await;
        if let Ok(admin) = PgPoolOptions::new()
            .max_connections(1)
            .connect(&self.admin_url)
            .await
        {
            let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                "DROP SCHEMA \"{}\" CASCADE",
                self.schema
            )))
            .execute(&admin)
            .await;
            admin.close().await;
        }
    }
}

async fn with_fixture<F, Fut>(name: &str, body: F)
where
    F: FnOnce(Fixture) -> Fut,
    Fut: std::future::Future<Output = Fixture>,
{
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; {name} pass-skipped");
        return;
    };
    let fx = body(fx).await;
    fx.cleanup().await;
}

fn conversation_with_messages(id: &str, count: usize) -> Conversation {
    let mut conv = Conversation::new(id, "persistence test");
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    for i in 0..count {
        let role = if i % 2 == 0 {
            Role::User
        } else {
            Role::Assistant
        };
        conv.messages
            .push(Message::new(role, format!("message {i}")));
    }
    conv
}

/// Row ids in ordinal order, fetched directly so the assertion is on the
/// actual persisted rows.
async fn message_ids(pool: &PgPool, conversation_id: &str) -> Vec<String> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT id FROM messages WHERE conversation_id = $1 ORDER BY ordinal")
            .bind(conversation_id)
            .fetch_all(pool)
            .await
            .expect("fetch message ids");
    rows.into_iter().map(|(id,)| id).collect()
}

async fn summary_ids_column(pool: &PgPool, conversation_id: &str) -> Vec<Option<String>> {
    let rows: Vec<(Option<String>,)> = sqlx::query_as(
        "SELECT summary_id FROM messages WHERE conversation_id = $1 ORDER BY ordinal",
    )
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .expect("fetch summary ids");
    rows.into_iter().map(|(s,)| s).collect()
}

// --- DS-5 acceptance criteria -------------------------------------------------

#[tokio::test]
async fn append_one_message_inserts_exactly_one_row_and_keeps_ids() {
    with_fixture(
        "append_one_message_inserts_exactly_one_row_and_keeps_ids",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let mut conv = conversation_with_messages("conv-append", 200);

            with_user_id(UserId::new("u1"), async {
                store.create(conv.clone()).await.expect("create");
            })
            .await;

            let before = message_ids(&fx.pool, "conv-append").await;
            assert_eq!(before.len(), 200);

            conv.messages
                .push(Message::new(Role::User, "the 201st message"));
            with_user_id(UserId::new("u1"), async {
                store.update(conv.clone()).await.expect("update");
            })
            .await;

            let after = message_ids(&fx.pool, "conv-append").await;
            assert_eq!(after.len(), 201, "exactly one row added");
            assert_eq!(
                &after[..200],
                &before[..],
                "the 200 pre-existing rows must keep their ids (no delete-all rewrite)"
            );
            assert!(
                !before.contains(&after[200]),
                "the appended row must be a new id"
            );

            fx
        },
    )
    .await;
}

#[tokio::test]
async fn message_ids_round_trip_through_get_and_are_ascending() {
    // #1: the persisted UUIDv7 id must round-trip onto each domain `Message`
    // via `get` (the new `msg_from_row` read path), be non-empty + distinct, and
    // already be in ascending order — so a client can take `max(id)` as a
    // high-water cursor for live subscription and back-paging.
    with_fixture(
        "message_ids_round_trip_through_get_and_are_ascending",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let conv = conversation_with_messages("conv-ids", 5);
            let created: Vec<String> = conv.messages.iter().map(|m| m.id.clone()).collect();

            with_user_id(UserId::new("u1"), async {
                store.create(conv.clone()).await.expect("create");
                let loaded = store
                    .get(&ConversationId::from("conv-ids"))
                    .await
                    .expect("get");
                let loaded_ids: Vec<String> =
                    loaded.messages.iter().map(|m| m.id.clone()).collect();
                assert_eq!(
                    loaded_ids, created,
                    "each message's persisted id must round-trip through get"
                );
                assert!(
                    loaded_ids.iter().all(|id| !id.is_empty()),
                    "no empty ids: {loaded_ids:?}"
                );
                let mut sorted = loaded_ids.clone();
                sorted.sort();
                assert_eq!(
                    sorted, loaded_ids,
                    "monotonic v7 ids must come back in ascending order"
                );
            })
            .await;

            fx
        },
    )
    .await;
}

#[tokio::test]
async fn noop_update_keeps_all_ids() {
    with_fixture("noop_update_keeps_all_ids", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let conv = conversation_with_messages("conv-noop", 20);

        with_user_id(UserId::new("u1"), async {
            store.create(conv.clone()).await.expect("create");
        })
        .await;
        let before = message_ids(&fx.pool, "conv-noop").await;

        with_user_id(UserId::new("u1"), async {
            store.update(conv.clone()).await.expect("update");
        })
        .await;
        let after = message_ids(&fx.pool, "conv-noop").await;

        assert_eq!(before, after, "no-op update must not rewrite any row");
        fx
    })
    .await;
}

#[tokio::test]
async fn summary_assignment_updates_rows_in_place() {
    with_fixture(
        "summary_assignment_updates_rows_in_place",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let mut conv = conversation_with_messages("conv-summary", 10);

            with_user_id(UserId::new("u1"), async {
                store.create(conv.clone()).await.expect("create");
            })
            .await;
            let before = message_ids(&fx.pool, "conv-summary").await;

            // Simulate compaction assigning a summary id to the first 4 messages
            // through the normal whole-conversation update path. (The summary row
            // itself isn't needed for the messages-table assertion.)
            for msg in conv.messages.iter_mut().take(4) {
                msg.summary_id = Some("summary-1".to_string());
            }
            // messages.summary_id has an FK to message_summaries — create the row.
            sqlx::query(
                "INSERT INTO message_summaries \
                (id, user_id, conversation_id, summary, start_ordinal, end_ordinal) \
             VALUES ('summary-1', 'u1', 'conv-summary', 's', 0, 3)",
            )
            .execute(&fx.pool)
            .await
            .expect("insert summary row");

            with_user_id(UserId::new("u1"), async {
                store.update(conv.clone()).await.expect("update");
            })
            .await;

            let after = message_ids(&fx.pool, "conv-summary").await;
            assert_eq!(before, after, "metadata-only change must keep all row ids");

            let summary_col = summary_ids_column(&fx.pool, "conv-summary").await;
            assert!(
                summary_col[..4]
                    .iter()
                    .all(|s| s.as_deref() == Some("summary-1")),
                "summary_id must be persisted on the first 4 rows: {summary_col:?}"
            );
            assert!(summary_col[4..].iter().all(|s| s.is_none()));

            fx
        },
    )
    .await;
}

#[tokio::test]
async fn tail_truncation_deletes_only_the_tail() {
    with_fixture("tail_truncation_deletes_only_the_tail", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let mut conv = conversation_with_messages("conv-trunc", 10);

        with_user_id(UserId::new("u1"), async {
            store.create(conv.clone()).await.expect("create");
        })
        .await;
        let before = message_ids(&fx.pool, "conv-trunc").await;

        conv.messages.truncate(6);
        with_user_id(UserId::new("u1"), async {
            store.update(conv.clone()).await.expect("update");
        })
        .await;

        let after = message_ids(&fx.pool, "conv-trunc").await;
        assert_eq!(after.len(), 6);
        assert_eq!(after[..], before[..6], "surviving rows keep their ids");

        fx
    })
    .await;
}

#[tokio::test]
async fn mid_history_removal_round_trips() {
    with_fixture("mid_history_removal_round_trips", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let mut conv = conversation_with_messages("conv-mid", 8);

        with_user_id(UserId::new("u1"), async {
            store.create(conv.clone()).await.expect("create");
        })
        .await;
        let before = message_ids(&fx.pool, "conv-mid").await;

        // Remove a message from the middle (compaction's trim_tool_pairs shape).
        conv.messages.remove(3);
        with_user_id(UserId::new("u1"), async {
            store.update(conv.clone()).await.expect("update");
        })
        .await;

        let after = message_ids(&fx.pool, "conv-mid").await;
        assert_eq!(after.len(), 7);
        assert_eq!(after[..3], before[..3], "prefix rows keep their ids");

        // Read-back must equal the in-memory conversation.
        let loaded = with_user_id(UserId::new("u1"), async {
            store.get(&ConversationId::from("conv-mid")).await
        })
        .await
        .expect("get");
        assert_eq!(loaded.messages.len(), 7);
        for (i, (got, want)) in loaded.messages.iter().zip(&conv.messages).enumerate() {
            assert_eq!(got.content, want.content, "content mismatch at index {i}");
            assert_eq!(got.role, want.role, "role mismatch at index {i}");
        }

        fx
    })
    .await;
}

#[tokio::test]
async fn update_round_trips_tool_calls_and_tool_results() {
    with_fixture(
        "update_round_trips_tool_calls_and_tool_results",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let mut conv = conversation_with_messages("conv-tools", 2);
            conv.messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    "call-1",
                    "read_file",
                    r#"{"path":"/tmp/x"}"#,
                )]));
            conv.messages
                .push(Message::tool_result("call-1", "file contents"));

            with_user_id(UserId::new("u1"), async {
                store.create(conv.clone()).await.expect("create");
            })
            .await;
            let before = message_ids(&fx.pool, "conv-tools").await;

            conv.messages
                .push(Message::new(Role::Assistant, "done reading"));
            with_user_id(UserId::new("u1"), async {
                store.update(conv.clone()).await.expect("update");
            })
            .await;

            let after = message_ids(&fx.pool, "conv-tools").await;
            assert_eq!(after.len(), 5);
            assert_eq!(after[..4], before[..], "existing rows keep their ids");

            let loaded = with_user_id(UserId::new("u1"), async {
                store.get(&ConversationId::from("conv-tools")).await
            })
            .await
            .expect("get");
            assert_eq!(loaded.messages.len(), 5);
            assert_eq!(loaded.messages[2].tool_calls.len(), 1);
            assert_eq!(loaded.messages[2].tool_calls[0].id, "call-1");
            assert_eq!(loaded.messages[2].tool_calls[0].name, "read_file");
            assert_eq!(loaded.messages[3].tool_call_id.as_deref(), Some("call-1"));
            assert_eq!(loaded.messages[3].content, "file contents");

            fx
        },
    )
    .await;
}

/// Direct read of the `idempotency_key` column in ordinal order, so an
/// assertion can distinguish a persisted SQL `NULL` from a loaded `None`.
async fn idempotency_key_column(pool: &PgPool, conversation_id: &str) -> Vec<Option<String>> {
    let rows: Vec<(Option<String>,)> = sqlx::query_as(
        "SELECT idempotency_key FROM messages WHERE conversation_id = $1 ORDER BY ordinal",
    )
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .expect("fetch idempotency keys");
    rows.into_iter().map(|(k,)| k).collect()
}

// --- #570 Phase 1b: idempotency-key persistence + load-surfacing ------------

/// A user message stamped with an idempotency key persists it, and `get`
/// surfaces the key back onto the domain `Message` so a reconnecting client can
/// dedupe by exact match instead of a content compare.
#[tokio::test]
async fn user_message_idempotency_key_round_trips_through_get() {
    with_fixture(
        "user_message_idempotency_key_round_trips_through_get",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let mut conv = conversation_with_messages("conv-idem", 0);
            let mut user_msg = Message::new(Role::User, "remember me");
            user_msg.idempotency_key = Some("idem-key-1".to_string());
            conv.messages.push(user_msg);

            with_user_id(UserId::new("u1"), async {
                store.create(conv.clone()).await.expect("create");
                let loaded = store
                    .get(&ConversationId::from("conv-idem"))
                    .await
                    .expect("get");
                assert_eq!(loaded.messages.len(), 1);
                assert_eq!(
                    loaded.messages[0].idempotency_key.as_deref(),
                    Some("idem-key-1"),
                    "the persisted user idempotency_key must round-trip through get"
                );
            })
            .await;

            assert_eq!(
                idempotency_key_column(&fx.pool, "conv-idem").await,
                vec![Some("idem-key-1".to_string())],
                "the key must be stored in the idempotency_key column"
            );

            fx
        },
    )
    .await;
}

/// Assistant rows never carry an idempotency key: only the user row that
/// initiated the turn does. The assistant row's column is a real SQL `NULL`.
#[tokio::test]
async fn assistant_message_idempotency_key_persists_null() {
    with_fixture(
        "assistant_message_idempotency_key_persists_null",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let mut conv = conversation_with_messages("conv-idem-asst", 0);
            let mut user_msg = Message::new(Role::User, "hello");
            user_msg.idempotency_key = Some("k-user".to_string());
            conv.messages.push(user_msg);
            // Assistant reply pushed later in a real turn — never keyed.
            conv.messages
                .push(Message::new(Role::Assistant, "hi back"));

            with_user_id(UserId::new("u1"), async {
                store.create(conv.clone()).await.expect("create");
                let loaded = store
                    .get(&ConversationId::from("conv-idem-asst"))
                    .await
                    .expect("get");
                assert_eq!(
                    loaded.messages[0].idempotency_key.as_deref(),
                    Some("k-user")
                );
                assert_eq!(
                    loaded.messages[1].idempotency_key, None,
                    "assistant rows never carry a client idempotency key"
                );
            })
            .await;

            assert_eq!(
                idempotency_key_column(&fx.pool, "conv-idem-asst").await,
                vec![Some("k-user".to_string()), None],
                "the assistant row's idempotency_key column must be SQL NULL"
            );

            fx
        },
    )
    .await;
}

/// A keyless user message (the pre-1b send path) persists no key and loads with
/// `idempotency_key == None`, so the column is backward-compatible.
#[tokio::test]
async fn user_message_without_key_loads_as_none() {
    with_fixture("user_message_without_key_loads_as_none", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let mut conv = conversation_with_messages("conv-idem-keyless", 0);
        // A plain user message: `idempotency_key` defaults to None.
        conv.messages
            .push(Message::new(Role::User, "no key here"));

        with_user_id(UserId::new("u1"), async {
            store.create(conv.clone()).await.expect("create");
            let loaded = store
                .get(&ConversationId::from("conv-idem-keyless"))
                .await
                .expect("get");
            assert_eq!(
                loaded.messages[0].idempotency_key, None,
                "a keyless user message loads with a None idempotency_key"
            );
        })
        .await;

        assert_eq!(
            idempotency_key_column(&fx.pool, "conv-idem-keyless").await,
            vec![None],
            "a keyless user message stores SQL NULL"
        );

        fx
    })
    .await;
}

#[tokio::test]
async fn cross_user_update_is_not_found_and_writes_nothing() {
    with_fixture(
        "cross_user_update_is_not_found_and_writes_nothing",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let mut conv = conversation_with_messages("conv-owned", 3);

            with_user_id(UserId::new("alice"), async {
                store.create(conv.clone()).await.expect("create");
            })
            .await;
            let before = message_ids(&fx.pool, "conv-owned").await;

            conv.messages.push(Message::new(Role::User, "bob was here"));
            let result = with_user_id(UserId::new("bob"), async {
                store.update(conv.clone()).await
            })
            .await;
            assert!(
                matches!(result, Err(CoreError::ConversationNotFound(_))),
                "cross-user update must be NotFound, got {result:?}"
            );

            let after = message_ids(&fx.pool, "conv-owned").await;
            assert_eq!(before, after, "bob's update must not touch alice's rows");

            fx
        },
    )
    .await;
}
