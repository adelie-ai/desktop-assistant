//! Integration coverage for `PgConversationSearchStore::search_messages`
//! (issue #437).
//!
//! `search_messages` is a hybrid over two axes — the message tsvector and the
//! conversation title/summary tsvector — and is double-scoped
//! (`m.user_id = $4 AND c.user_id = $4`) as defense-in-depth. These tests pin
//! that both axes work and that each `user_id` filter matters, including a
//! deliberately corrupted row (a message stamped with a different user than its
//! conversation) that the `m.user_id` half must hide even though the JOIN would
//! otherwise expose it.
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::Arc;

use desktop_assistant_core::domain::{Conversation, Message, Role};
use desktop_assistant_core::ports::conversation_search::ConversationSearchStore;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{
    PgConversationSearchStore, PgConversationStore, UserId, run_migrations, with_user_id,
};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("issue437cs_{}", Uuid::now_v7().simple());

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

fn conversation(id: &str, title: &str, message: &str) -> Conversation {
    let mut conv = Conversation::new(id, title);
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, message));
    conv
}

#[tokio::test]
async fn search_messages_is_double_user_scoped() {
    // Covers both search axes (message FTS and conversation title/summary) and
    // both halves of the `(m.user_id, c.user_id)` scope.
    //
    // MUTATION (central): dropping `m.user_id = $4` lets Alice's "zebra" search
    // match the corrupted row (a `bob`-owned message living in Alice's
    // conversation) → RED. Dropping the whole scope additionally leaks Bob's
    // legitimate rows into Alice's results.
    with_fixture("search_messages_is_double_user_scoped", |fx| async move {
        let conv_store = PgConversationStore::new(fx.pool.clone());
        let search = PgConversationSearchStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            conv_store
                .create(conversation(
                    "conv-a",
                    "Alpha planning",
                    "discuss the widget rollout",
                ))
                .await
                .expect("alice create");
        })
        .await;
        with_user_id(UserId::new("bob"), async {
            conv_store
                .create(conversation(
                    "conv-b",
                    "Beta planning",
                    "discuss the widget rollout",
                ))
                .await
                .expect("bob create");
        })
        .await;

        // (a) message-FTS axis, scoped: Alice's "widget" search sees only her
        // conversation, never Bob's identical message.
        let alice_widget = with_user_id(UserId::new("alice"), async {
            search.search_messages("widget", 20, None).await
        })
        .await
        .expect("alice widget search");
        assert!(
            !alice_widget.is_empty(),
            "alice must find her own 'widget' message"
        );
        assert!(
            alice_widget.iter().all(|h| h.conversation_id == "conv-a"),
            "alice's message search must stay in her conversations; got {:?}",
            alice_widget
                .iter()
                .map(|h| &h.conversation_id)
                .collect::<Vec<_>>()
        );

        // (b) title/summary axis, scoped: a token only in the TITLE surfaces for
        // the owner and is invisible cross-user.
        let alice_title = with_user_id(UserId::new("alice"), async {
            search.search_messages("Alpha", 20, None).await
        })
        .await
        .expect("alice title search");
        assert!(
            alice_title.iter().any(|h| h.conversation_id == "conv-a"),
            "alice must find her conversation by a title-only term"
        );
        let bob_title = with_user_id(UserId::new("bob"), async {
            search.search_messages("Alpha", 20, None).await
        })
        .await
        .expect("bob title search");
        assert!(
            bob_title.is_empty(),
            "bob must NOT see alice's conversation via the title axis; got {:?}",
            bob_title
                .iter()
                .map(|h| &h.conversation_id)
                .collect::<Vec<_>>()
        );

        // (c) corrupted row: a message stamped user_id='bob' but living inside
        // Alice's conversation. The `m.user_id = $4` half must hide it from
        // Alice; the `c.user_id = $4` half hides it from Bob.
        sqlx::query(
            "INSERT INTO messages (id, user_id, conversation_id, ordinal, role, content) \
             VALUES ($1, 'bob', 'conv-a', 50, 'user', 'zebra crossing ahead')",
        )
        .bind(Uuid::now_v7().to_string())
        .execute(&fx.pool)
        .await
        .expect("insert corrupt row");

        let alice_zebra = with_user_id(UserId::new("alice"), async {
            search.search_messages("zebra", 20, None).await
        })
        .await
        .expect("alice zebra search");
        assert!(
            alice_zebra.is_empty(),
            "the `m.user_id` scope must hide a foreign-owned message even inside \
             alice's own conversation; got {:?}",
            alice_zebra
                .iter()
                .map(|h| &h.conversation_id)
                .collect::<Vec<_>>()
        );

        let bob_zebra = with_user_id(UserId::new("bob"), async {
            search.search_messages("zebra", 20, None).await
        })
        .await
        .expect("bob zebra search");
        assert!(
            bob_zebra.is_empty(),
            "the `c.user_id` scope must hide a message whose conversation belongs \
             to another user; got {:?}",
            bob_zebra
                .iter()
                .map(|h| &h.conversation_id)
                .collect::<Vec<_>>()
        );
        fx
    })
    .await;
}
