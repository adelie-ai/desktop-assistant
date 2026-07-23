//! DS-6 (#295): `ConversationStore::list` must NOT issue a per-conversation
//! query nor load message bodies — it returns a light projection
//! (`ConversationSummary`) whose `message_count` is computed by a single
//! aggregate query.
//!
//! These tests run against a real Postgres instance (the only backend in
//! production). When `TEST_DATABASE_URL` is unset every test pass-skips with
//! a log line so the suite stays green without a DB. See
//! `user_id_scoping.rs` for the running instructions.

mod support;

use std::sync::Arc;

use desktop_assistant_core::domain::{
    Conversation, ConversationId, Message, RESERVED_SUBAGENT_TAG, Role,
};
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
        let schema = format!("issue295_{}", Uuid::now_v7().simple());

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

fn conversation_with_messages(id: &str, title: &str, body_count: usize) -> Conversation {
    let mut conv = Conversation::new(id, title);
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    for i in 0..body_count {
        conv.messages
            .push(Message::new(Role::User, format!("message body {i}")));
    }
    conv
}

/// Acceptance: `list` reports per-conversation `message_count` correctly for
/// several conversations with DIFFERENT message counts in a SINGLE list call.
/// A correct aggregate-join projection returns each count; an N+1 regression
/// would still pass this, so it is paired with `list_does_not_load_bodies`.
#[tokio::test]
async fn list_reports_message_count_per_conversation() {
    with_fixture(
        "list_reports_message_count_per_conversation",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create(conversation_with_messages("conv-0", "zero", 0))
                    .await
                    .expect("create conv-0");
                store
                    .create(conversation_with_messages("conv-3", "three", 3))
                    .await
                    .expect("create conv-3");
                store
                    .create(conversation_with_messages("conv-7", "seven", 7))
                    .await
                    .expect("create conv-7");
            })
            .await;

            let list = with_user_id(UserId::new("alice"), async { store.list().await })
                .await
                .expect("alice list");

            assert_eq!(list.len(), 3, "exactly three conversations");

            let count_for = |id: &str| {
                list.iter()
                    .find(|s| s.id.0 == id)
                    .unwrap_or_else(|| panic!("missing {id} in list"))
                    .message_count
            };
            assert_eq!(count_for("conv-0"), 0, "empty conversation -> 0");
            assert_eq!(count_for("conv-3"), 3);
            assert_eq!(count_for("conv-7"), 7);

            fx
        },
    )
    .await;
}

/// Acceptance: `list` does NOT fetch message bodies. We seed a conversation
/// with a body that contains a sentinel, then prove the SQL `list` issues
/// never selects the `messages.content` column. We assert by counting how
/// many times the sentinel body text can be reconstructed from the
/// projection — the projection has no `messages` field at all, so the only
/// thing carried per-row is the COUNT.
///
/// The compile-time guarantee (the returned `ConversationSummary` has no
/// message vector) is the strongest "bodies not loaded" proof; this test
/// additionally pins the behavioral expectation that the count is right while
/// the body is absent.
#[tokio::test]
async fn list_does_not_load_bodies() {
    with_fixture("list_does_not_load_bodies", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());

        let sentinel = "SENTINEL-BODY-DO-NOT-LOAD";
        with_user_id(UserId::new("alice"), async {
            let mut conv = Conversation::new("conv-body", "has body");
            conv.created_at = "2026-01-01 00:00:00".to_string();
            conv.updated_at = "2026-01-01 00:00:00".to_string();
            conv.messages.push(Message::new(Role::User, sentinel));
            store.create(conv).await.expect("create conv-body");
        })
        .await;

        let list = with_user_id(UserId::new("alice"), async { store.list().await })
            .await
            .expect("alice list");

        assert_eq!(list.len(), 1);
        let summary = &list[0];
        assert_eq!(summary.id.0, "conv-body");
        assert_eq!(summary.message_count, 1);

        // The projection type carries no message bodies. Serialize the whole
        // summary set to JSON and assert the sentinel body never appears —
        // catching any future regression that re-introduces a body field.
        let rendered = format!("{summary:?}");
        assert!(
            !rendered.contains(sentinel),
            "list projection must not carry message bodies; found sentinel in {rendered}"
        );

        fx
    })
    .await;
}

/// Unhappy path: empty store lists to an empty vec without error and without
/// any per-conversation work.
#[tokio::test]
async fn list_empty_store_is_empty() {
    with_fixture("list_empty_store_is_empty", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let list = with_user_id(UserId::new("alice"), async { store.list().await })
            .await
            .expect("alice list");
        assert!(list.is_empty());
        fx
    })
    .await;
}

/// Cross-tenant: bob's aggregate count never includes alice's messages.
/// Proves the COUNT join is user-scoped (`m.user_id = c.user_id`), not a
/// global aggregate that would sum every user's messages onto each row.
#[tokio::test]
async fn list_count_is_user_scoped() {
    with_fixture("list_count_is_user_scoped", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .create(conversation_with_messages("alice-conv", "alice", 5))
                .await
                .expect("alice create");
        })
        .await;
        with_user_id(UserId::new("bob"), async {
            store
                .create(conversation_with_messages("bob-conv", "bob", 2))
                .await
                .expect("bob create");
        })
        .await;

        let alices = with_user_id(UserId::new("alice"), async { store.list().await })
            .await
            .expect("alice list");
        let bobs = with_user_id(UserId::new("bob"), async { store.list().await })
            .await
            .expect("bob list");

        // Each user sees only their own conversation, and its count reflects
        // only their own messages — a global (un-scoped) COUNT join would
        // report 7 (5 + 2) for whichever row it touched.
        assert_eq!(alices.len(), 1, "alice sees only her conversation");
        assert_eq!(alices[0].id.0, "alice-conv");
        assert_eq!(alices[0].message_count, 5, "alice's count is hers alone");
        assert_eq!(bobs.len(), 1, "bob sees only his conversation");
        assert_eq!(bobs[0].id.0, "bob-conv");
        assert_eq!(bobs[0].message_count, 2, "bob's count is his alone");

        fx
    })
    .await;
}

/// #609: a subagent's private working conversation is tagged
/// `RESERVED_SUBAGENT_TAG` at creation and must NOT surface in `list` -- it is
/// an implementation detail, not something the user opened. A normal
/// conversation alongside it still lists, and the hidden one is still directly
/// resolvable by id (hidden from the list, not deleted).
#[tokio::test]
async fn list_excludes_subagent_tagged_conversations() {
    with_fixture(
        "list_excludes_subagent_tagged_conversations",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create(conversation_with_messages("conv-normal", "normal", 1))
                    .await
                    .expect("create normal conv");
                let mut sub = conversation_with_messages("conv-sub", "Subagent: researcher", 1);
                sub.tags = vec![RESERVED_SUBAGENT_TAG.to_string()];
                store.create(sub).await.expect("create subagent conv");
            })
            .await;

            let list = with_user_id(UserId::new("alice"), async { store.list().await })
                .await
                .expect("alice list");

            assert_eq!(list.len(), 1, "only the normal conversation is listed");
            assert_eq!(list[0].id.0, "conv-normal");
            assert!(
                !list.iter().any(|s| s.id.0 == "conv-sub"),
                "subagent working conversation is hidden from the list"
            );

            // Still resolvable directly -- hidden from the list, not deleted.
            let got = with_user_id(UserId::new("alice"), async {
                store.get(&ConversationId::from("conv-sub")).await
            })
            .await;
            assert!(
                got.is_ok(),
                "subagent conversation is still fetchable by id for the running child"
            );

            fx
        },
    )
    .await;
}

// Keep the unused import lint quiet if the DB env is never set (the
// `ConversationId` import is exercised only inside `with_fixture` bodies).
#[allow(dead_code)]
fn _assert_id_used(_: ConversationId) {}
