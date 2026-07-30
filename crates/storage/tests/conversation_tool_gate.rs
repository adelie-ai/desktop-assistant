//! Persistence tests for the per-conversation tool-provenance-gate override
//! (issue #1007): `PgConversationStore::{get,set}_conversation_tool_gate_disabled`.
//!
//! Mirrors `get_conversation_model_selection` / `get_conversation_personality`:
//! a plain user-scoped column, `ConversationNotFound` on a missing or
//! cross-user row (#105 existence-leak rule), and — because this column backs
//! a security-relevant gate — a fail-closed default that a caller one layer up
//! (the daemon's `resolve_tool_gate_disabled`) maps any error onto `false`.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test conversation_tool_gate
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{PgConversationStore, UserId, run_migrations, with_user_id};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// RAII fixture: private schema, pool pinned to it, migrations applied.
/// Mirrors `conversation_persistence.rs`'s `Fixture`.
struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("gate1007_{}", Uuid::now_v7().simple());

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

        let schema_for_hook = std::sync::Arc::new(schema.clone());
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .after_connect(move |conn, _| {
                let schema = std::sync::Arc::clone(&schema_for_hook);
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

#[tokio::test]
async fn a_freshly_created_conversation_has_the_gate_enforced() {
    with_fixture(
        "a_freshly_created_conversation_has_the_gate_enforced",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let conv = Conversation::new("conv-gate-default", "t");

            with_user_id(UserId::new("u1"), async {
                store.create(conv).await.expect("create");
                let disabled = store
                    .get_conversation_tool_gate_disabled(&ConversationId::from("conv-gate-default"))
                    .await
                    .expect("get");
                assert!(
                    !disabled,
                    "a conversation with no override must read as gate-enforced"
                );
            })
            .await;

            fx
        },
    )
    .await;
}

#[tokio::test]
async fn set_true_then_false_round_trips() {
    with_fixture("set_true_then_false_round_trips", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let conv = Conversation::new("conv-gate-roundtrip", "t");
        let id = ConversationId::from("conv-gate-roundtrip");

        with_user_id(UserId::new("u1"), async {
            store.create(conv).await.expect("create");

            store
                .set_conversation_tool_gate_disabled(&id, true)
                .await
                .expect("set true");
            assert!(
                store
                    .get_conversation_tool_gate_disabled(&id)
                    .await
                    .expect("get after set true")
            );

            store
                .set_conversation_tool_gate_disabled(&id, false)
                .await
                .expect("set false");
            assert!(
                !store
                    .get_conversation_tool_gate_disabled(&id)
                    .await
                    .expect("get after set false")
            );
        })
        .await;

        fx
    })
    .await;
}

#[tokio::test]
async fn set_on_an_unknown_conversation_is_not_found() {
    with_fixture(
        "set_on_an_unknown_conversation_is_not_found",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let result = with_user_id(UserId::new("u1"), async {
                store
                    .set_conversation_tool_gate_disabled(
                        &ConversationId::from("does-not-exist"),
                        true,
                    )
                    .await
            })
            .await;
            assert!(
                matches!(result, Err(CoreError::ConversationNotFound(_))),
                "got {result:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn get_on_an_unknown_conversation_is_not_found() {
    with_fixture(
        "get_on_an_unknown_conversation_is_not_found",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let result = with_user_id(UserId::new("u1"), async {
                store
                    .get_conversation_tool_gate_disabled(&ConversationId::from("does-not-exist"))
                    .await
            })
            .await;
            assert!(
                matches!(result, Err(CoreError::ConversationNotFound(_))),
                "got {result:?}"
            );
            fx
        },
    )
    .await;
}

/// #105: a cross-user probe must not distinguish "doesn't exist" from "not
/// yours" — both resolve to the same `ConversationNotFound`, and critically,
/// bob must not be able to flip alice's gate off by guessing her conversation
/// id.
#[tokio::test]
async fn cross_user_get_and_set_are_not_found_and_do_not_leak_or_mutate() {
    with_fixture(
        "cross_user_get_and_set_are_not_found_and_do_not_leak_or_mutate",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let conv = Conversation::new("conv-owned-by-alice", "t");
            let id = ConversationId::from("conv-owned-by-alice");

            with_user_id(UserId::new("alice"), async {
                store.create(conv).await.expect("create");
            })
            .await;

            let get_result = with_user_id(UserId::new("bob"), async {
                store.get_conversation_tool_gate_disabled(&id).await
            })
            .await;
            assert!(
                matches!(get_result, Err(CoreError::ConversationNotFound(_))),
                "bob's get must not see alice's conversation, got {get_result:?}"
            );

            let set_result = with_user_id(UserId::new("bob"), async {
                store.set_conversation_tool_gate_disabled(&id, true).await
            })
            .await;
            assert!(
                matches!(set_result, Err(CoreError::ConversationNotFound(_))),
                "bob's set must not mutate alice's conversation, got {set_result:?}"
            );

            // Alice's row must still read as gate-enforced: bob's refused
            // write must not have gone through.
            let alice_value = with_user_id(UserId::new("alice"), async {
                store.get_conversation_tool_gate_disabled(&id).await
            })
            .await
            .expect("alice can still read her own conversation");
            assert!(
                !alice_value,
                "bob's refused write must not have flipped alice's gate"
            );

            fx
        },
    )
    .await;
}
