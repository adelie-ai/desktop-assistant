//! Integration tests for the SendMessage idempotency-key store (#204).
//!
//! Exercises `PgIdempotencyKeyStore` end-to-end against a real Postgres with
//! the migration applied: record + lookup roundtrip, absent-key miss,
//! idempotent upsert, per-(user, conversation, key) scoping, and
//! cascade-delete with the parent conversation.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test idempotency_keys
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log line so
//! the suite stays green without a DB.

mod support;

use std::sync::Arc;

use desktop_assistant_core::domain::{Conversation, ConversationId, Message, Role};
use desktop_assistant_core::ports::store::{ConversationStore, IdempotencyKeyStore};
use desktop_assistant_storage::{
    PgConversationStore, PgIdempotencyKeyStore, UserId, run_migrations, with_user_id,
};
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
        let schema = format!("issue204_{}", Uuid::now_v7().simple());

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
            .max_connections(4)
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

fn make_conversation(id: &str) -> Conversation {
    let mut conv = Conversation::new(id, "idempotency test");
    conv.created_at = "2026-06-05 00:00:00".to_string();
    conv.updated_at = "2026-06-05 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

#[tokio::test]
async fn record_then_lookup_roundtrips() {
    with_fixture("record_then_lookup_roundtrips", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let store = PgIdempotencyKeyStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");

            assert_eq!(
                store.lookup_completed("c1", "k1").await.unwrap(),
                None,
                "an absent key must miss"
            );

            store
                .record_response("c1", "k1", "req-1", "the answer")
                .await
                .expect("record");

            assert_eq!(
                store.lookup_completed("c1", "k1").await.unwrap().as_deref(),
                Some("the answer"),
                "a recorded reply round-trips"
            );
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn record_is_idempotent_upsert() {
    with_fixture("record_is_idempotent_upsert", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let store = PgIdempotencyKeyStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");

            store
                .record_response("c1", "k1", "req-1", "first")
                .await
                .unwrap();
            // Re-recording the same key overwrites instead of erroring or
            // duplicating (a turn that raced past the dedup check and ran
            // twice still converges to one row).
            store
                .record_response("c1", "k1", "req-2", "second")
                .await
                .unwrap();

            assert_eq!(
                store.lookup_completed("c1", "k1").await.unwrap().as_deref(),
                Some("second"),
                "re-recording overwrites the stored reply"
            );
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn lookup_is_scoped_per_user() {
    with_fixture("lookup_is_scoped_per_user", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let store = PgIdempotencyKeyStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("alice conv");
            store
                .record_response("c1", "k1", "r", "alice-secret")
                .await
                .unwrap();
        })
        .await;

        with_user_id(UserId::new("bob"), async {
            assert_eq!(
                store.lookup_completed("c1", "k1").await.unwrap(),
                None,
                "another user's identical key must not be visible"
            );
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn rows_cascade_delete_with_conversation() {
    with_fixture("rows_cascade_delete_with_conversation", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let store = PgIdempotencyKeyStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            store
                .record_response("c1", "k1", "r", "answer")
                .await
                .unwrap();

            convs
                .delete(&ConversationId::from("c1"))
                .await
                .expect("delete conv");

            assert_eq!(
                store.lookup_completed("c1", "k1").await.unwrap(),
                None,
                "the idempotency row is cascade-deleted with its conversation"
            );
        })
        .await;
        fx
    })
    .await;
}
