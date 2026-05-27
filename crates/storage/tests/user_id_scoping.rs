//! Cross-user isolation integration tests for #105.
//!
//! These tests exercise the storage adapters end-to-end against a real
//! Postgres instance with the multi-tenant schema in place. They prove
//! that:
//!
//! - Writes under user A land in A's partition only.
//! - Reads under user A see only A's rows.
//! - Cross-user lookups return `ConversationNotFound` rather than
//!   leaking the existence of another user's data.
//! - The single-tenant default path (no `UserId` scope installed)
//!   resolves to the schema sentinel `"default"`, so an unscoped
//!   request can still write and read.
//!
//! ## Running locally
//!
//! Set `TEST_DATABASE_URL` to a Postgres URL where the connecting role
//! can `CREATE SCHEMA` and `CREATE EXTENSION vector`:
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test user_id_scoping
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log
//! line so the suite stays green without a DB.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, KnowledgeEntry, Message, Role};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{
    PgConversationStore, PgKnowledgeBaseStore, UserId, run_migrations, with_user_id,
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
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let schema = format!("issue105_{}", Uuid::now_v7().simple());

        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to TEST_DATABASE_URL");
        sqlx::query(&format!("CREATE SCHEMA \"{schema}\""))
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
                    sqlx::query(&sql).execute(conn).await?;
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
            let _ = sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", self.schema))
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

fn make_conversation(id: &str, title: &str) -> Conversation {
    let mut conv = Conversation::new(id, title);
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

// -- conversations -----------------------------------------------------------

#[tokio::test]
async fn user_a_writes_then_user_b_cannot_list_it() {
    // Two users concurrent: A writes a conversation under their scope;
    // B's list under their own scope returns nothing. Mirrors the
    // top-level `two_users_concurrent_conversations_isolated` test in
    // the issue brief.
    with_fixture("user_a_writes_then_user_b_cannot_list_it", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());

        // User A: insert a conversation.
        with_user_id(UserId::new("alice"), async {
            store
                .create(make_conversation("conv-a-1", "alice chat"))
                .await
                .expect("alice create");
        })
        .await;

        // User B: list — should see nothing.
        let bobs_list = with_user_id(UserId::new("bob"), async { store.list().await })
            .await
            .expect("bob list");
        assert!(
            bobs_list.is_empty(),
            "bob's list must NOT include alice's conversations, got {} row(s)",
            bobs_list.len()
        );

        // User A: list — should see exactly their one row.
        let alices_list = with_user_id(UserId::new("alice"), async { store.list().await })
            .await
            .expect("alice list");
        assert_eq!(alices_list.len(), 1);
        assert_eq!(alices_list[0].id.0, "conv-a-1");

        fx
    })
    .await;
}

#[tokio::test]
async fn user_b_cannot_read_user_a_conversation_by_id() {
    // Cross-user direct lookup must return NotFound — don't leak the
    // existence of A's conversation to B.
    with_fixture("user_b_cannot_read_user_a_conversation_by_id", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .create(make_conversation("conv-x", "alice"))
                .await
                .expect("alice create");
        })
        .await;

        let bob_get =
            with_user_id(UserId::new("bob"), async { store.get(&ConversationId::from("conv-x")).await }).await;
        match bob_get {
            Err(CoreError::ConversationNotFound(id)) => {
                assert_eq!(id, "conv-x", "bob's NotFound must echo the id he asked for");
            }
            other => panic!("expected ConversationNotFound, got {other:?}"),
        }
        fx
    })
    .await;
}

#[tokio::test]
async fn user_b_cannot_delete_user_a_conversation_by_id() {
    // Mutating cross-user attempt also returns NotFound — same don't-
    // leak-existence rule as the read path.
    with_fixture("user_b_cannot_delete_user_a_conversation_by_id", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .create(make_conversation("conv-y", "alice"))
                .await
                .expect("alice create");
        })
        .await;

        let bob_delete = with_user_id(UserId::new("bob"), async {
            store.delete(&ConversationId::from("conv-y")).await
        })
        .await;
        match bob_delete {
            Err(CoreError::ConversationNotFound(id)) => {
                assert_eq!(id, "conv-y");
            }
            other => panic!("expected ConversationNotFound, got {other:?}"),
        }

        // Alice's row is still intact.
        let alices_get = with_user_id(UserId::new("alice"), async {
            store.get(&ConversationId::from("conv-y")).await
        })
        .await;
        assert!(alices_get.is_ok(), "alice's row should still be readable");
        fx
    })
    .await;
}

#[tokio::test]
async fn two_users_concurrent_conversations_isolated() {
    // tokio::join concurrent writes — neither user sees the other's
    // data afterwards. Direct exercise of the issue's primary
    // acceptance test.
    with_fixture("two_users_concurrent_conversations_isolated", |fx| async move {
        let store = Arc::new(PgConversationStore::new(fx.pool.clone()));

        let store_a = Arc::clone(&store);
        let store_b = Arc::clone(&store);

        let alice = with_user_id(UserId::new("alice"), async move {
            store_a.create(make_conversation("conv-a", "alice")).await
        });
        let bob = with_user_id(UserId::new("bob"), async move {
            store_b.create(make_conversation("conv-b", "bob")).await
        });
        let (a, b) = tokio::join!(alice, bob);
        a.expect("alice create");
        b.expect("bob create");

        let alices_list =
            with_user_id(UserId::new("alice"), async { store.list().await }).await.unwrap();
        let bobs_list =
            with_user_id(UserId::new("bob"), async { store.list().await }).await.unwrap();

        assert_eq!(alices_list.len(), 1);
        assert_eq!(alices_list[0].id.0, "conv-a");
        assert_eq!(bobs_list.len(), 1);
        assert_eq!(bobs_list[0].id.0, "conv-b");
        fx
    })
    .await;
}

#[tokio::test]
async fn existing_messages_table_inserts_set_user_id() {
    // Regression: prove the message-insert path stamps the
    // task-local user id into messages.user_id (not just
    // conversations.user_id).
    with_fixture("existing_messages_table_inserts_set_user_id", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .create(make_conversation("conv-msg", "alice"))
                .await
                .expect("alice create");
        })
        .await;

        let row: (String,) = sqlx::query_as(
            "SELECT user_id FROM messages WHERE conversation_id = $1 LIMIT 1",
        )
        .bind("conv-msg")
        .fetch_one(&fx.pool)
        .await
        .expect("read back message row");
        assert_eq!(row.0, "alice");
        fx
    })
    .await;
}

#[tokio::test]
async fn request_with_default_user_id_succeeds_in_single_tenant_path() {
    // Single-tenant deploy boundary: no scope installed, no JWT in
    // play, write/read still works against the sentinel partition.
    with_fixture(
        "request_with_default_user_id_succeeds_in_single_tenant_path",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            // No `with_user_id` wrapper — the storage layer falls
            // through to the sentinel.
            store
                .create(make_conversation("conv-default", "single tenant"))
                .await
                .expect("create with no scope installed");

            let list = store.list().await.expect("list with no scope");
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].id.0, "conv-default");

            // The persisted row must carry the sentinel `user_id`.
            let row: (String,) = sqlx::query_as(
                "SELECT user_id FROM conversations WHERE id = $1",
            )
            .bind("conv-default")
            .fetch_one(&fx.pool)
            .await
            .expect("read back");
            assert_eq!(row.0, "default");
            fx
        },
    )
    .await;
}

// -- knowledge base ----------------------------------------------------------

#[tokio::test]
async fn knowledge_writes_are_isolated_per_user() {
    with_fixture("knowledge_writes_are_isolated_per_user", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            let entry = KnowledgeEntry::new("kb-alice", "alice loves rust", vec!["pref".into()]);
            store.write(entry, None, None).await.expect("alice write");
        })
        .await;

        with_user_id(UserId::new("bob"), async {
            let entry = KnowledgeEntry::new("kb-bob", "bob loves zig", vec!["pref".into()]);
            store.write(entry, None, None).await.expect("bob write");
        })
        .await;

        // Each user lists only their own entry.
        let alices = with_user_id(UserId::new("alice"), async {
            store.list(100, 0, None).await
        })
        .await
        .unwrap();
        assert_eq!(alices.len(), 1);
        assert_eq!(alices[0].id, "kb-alice");

        let bobs = with_user_id(UserId::new("bob"), async {
            store.list(100, 0, None).await
        })
        .await
        .unwrap();
        assert_eq!(bobs.len(), 1);
        assert_eq!(bobs[0].id, "kb-bob");
        fx
    })
    .await;
}

#[tokio::test]
async fn knowledge_search_does_not_leak_across_users() {
    // pgvector / FTS path: user A indexes a doc; user B searches the
    // same query and doesn't see it. Mirrors the issue brief's
    // `knowledge_search_does_not_leak_across_users` test.
    with_fixture("knowledge_search_does_not_leak_across_users", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            let entry = KnowledgeEntry::new(
                "kb-alice-doc",
                "Alice's private project notes about widget refactoring",
                vec!["project".into()],
            );
            store.write(entry, None, None).await.expect("alice index");
        })
        .await;

        // FTS-only path (no embedding required).
        let bob_hits = with_user_id(UserId::new("bob"), async {
            store.search_text("widget refactoring", None, 10).await
        })
        .await
        .unwrap();
        assert!(
            bob_hits.is_empty(),
            "bob's text search must not see alice's doc, got {} hit(s)",
            bob_hits.len()
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn knowledge_get_by_id_does_not_leak_across_users() {
    with_fixture("knowledge_get_by_id_does_not_leak_across_users", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            let entry =
                KnowledgeEntry::new("kb-shared-id", "alice content", vec!["pref".into()]);
            store.write(entry, None, None).await.expect("alice write");
        })
        .await;

        // Bob tries to fetch by the exact same id — must miss.
        let bob_fetch = with_user_id(UserId::new("bob"), async { store.get("kb-shared-id").await })
            .await
            .unwrap();
        assert!(
            bob_fetch.is_none(),
            "bob's get by id must miss when the id belongs to alice"
        );

        // Alice still sees it.
        let alice_fetch =
            with_user_id(UserId::new("alice"), async { store.get("kb-shared-id").await })
                .await
                .unwrap();
        assert!(alice_fetch.is_some());
        fx
    })
    .await;
}
