//! Integration tests for the tool-usage cost aggregate (#599).
//!
//! Exercises `PgToolUsageStore` against a real Postgres with the migrations
//! applied. The SQL is the whole feature here — the aggregate is derived, so
//! there is no Rust logic to unit-test in its place; these tests are the spec.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test tool_usage
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log line so
//! the suite stays green without a DB.

mod support;

use std::sync::Arc;

use desktop_assistant_core::domain::{Conversation, Message, Role, ToolCall};
use desktop_assistant_core::planning::COMPACTION_POINTER_PREFIX;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_core::ports::tool_usage::{ToolUsage, ToolUsageStore};
use desktop_assistant_storage::tool_usage::PgToolUsageStore;
use desktop_assistant_storage::{PgConversationStore, UserId, run_migrations, with_user_id};
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
        let schema = format!("issue599_{}", Uuid::now_v7().simple());
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect admin pool");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA \"{schema}\"")))
            .execute(&admin)
            .await
            .expect("create schema");
        admin.close().await;

        let pool = PgPoolOptions::new()
            .max_connections(2)
            .after_connect({
                let schema = Arc::new(schema.clone());
                move |conn, _| {
                    let schema = Arc::clone(&schema);
                    Box::pin(async move {
                        let sql = format!("SET search_path TO \"{schema}\", public");
                        sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                        Ok(())
                    })
                }
            })
            .connect(&url)
            .await
            .expect("connect scoped pool");
        run_migrations(&pool).await.expect("migrations");
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

/// Persist a conversation whose assistant messages request `calls` and whose
/// tool messages carry `results`, so the aggregate has something to chew on.
async fn seed(
    pool: &PgPool,
    conv_id: &str,
    calls: &[(&str, &str)],
    results: &[(&str, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    let store = PgConversationStore::new(pool.clone());
    let mut conv = Conversation::new(conv_id, "usage");
    conv.created_at = "2026-06-03 00:00:00".to_string();
    conv.updated_at = "2026-06-03 00:00:00".to_string();
    for (call_id, name) in calls {
        let mut m = Message::new(Role::Assistant, String::new());
        m.tool_calls = vec![ToolCall::new(*call_id, *name, "{}")];
        conv.messages.push(m);
    }
    for (call_id, body) in results {
        let mut m = Message::new(Role::Tool, body.clone());
        m.tool_call_id = Some((*call_id).to_string());
        conv.messages.push(m);
    }
    store.create(conv).await?;
    Ok(())
}

fn by_name<'a>(rows: &'a [ToolUsage], name: &str) -> &'a ToolUsage {
    rows.iter()
        .find(|r| r.tool_name == name)
        .unwrap_or_else(|| panic!("no usage row for {name}; got {rows:?}"))
}

#[tokio::test]
async fn tool_usage_counts_per_tool() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        seed(
            &fx.pool,
            "c1",
            &[
                ("a1", "search"),
                ("a2", "search"),
                ("a3", "search"),
                ("b1", "fetch"),
            ],
            &[],
        )
        .await
        .expect("seed");
        let rows = store.tool_usage("c1").await.expect("aggregate");
        assert_eq!(rows.len(), 2, "one row per distinct tool: {rows:?}");
        // Ordered by count desc, so the busiest tool leads.
        assert_eq!(rows[0].tool_name, "search");
        assert_eq!(rows[0].call_count, 3);
        assert_eq!(rows[1].tool_name, "fetch");
        assert_eq!(rows[1].call_count, 1);
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_usage_reports_payload_bytes_and_tokens() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        seed(
            &fx.pool,
            "c1",
            &[("a1", "fetch"), ("a2", "fetch")],
            &[("a1", "x".repeat(400)), ("a2", "y".repeat(600))],
        )
        .await
        .expect("seed");
        let rows = store.tool_usage("c1").await.expect("aggregate");
        let fetch = by_name(&rows, "fetch");
        assert_eq!(fetch.result_bytes, 1000, "bytes are summed across results");
        assert_eq!(
            fetch.result_tokens(),
            250,
            "tokens use the shared bytes/4 rule"
        );
        assert_eq!(fetch.max_result_bytes, 600, "largest single result");
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_usage_max_result_bytes_flags_a_single_large_dump() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        // The case a count-ordered chart gets backwards: `chatty` is called far
        // more often, but `heavy` is what actually ate the context.
        let mut calls: Vec<(&str, &str)> = (0..10).map(|_| ("c", "chatty")).collect();
        calls.push(("h1", "heavy"));
        let results = vec![("h1", "z".repeat(40_000))];
        seed(&fx.pool, "c1", &calls, &results).await.expect("seed");

        let rows = store.tool_usage("c1").await.expect("aggregate");
        let heavy = by_name(&rows, "heavy");
        let chatty = by_name(&rows, "chatty");
        assert!(
            chatty.call_count > heavy.call_count,
            "chatty is called more"
        );
        assert!(
            heavy.max_result_bytes > chatty.max_result_bytes,
            "but heavy is the expensive one — the payload axis must show it"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_usage_survives_compaction() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        // An evicted result: content replaced by the compaction pointer (#240).
        let evicted = format!("{COMPACTION_POINTER_PREFIX} 3-5: distilled into note(s) \"x\".>");
        seed(
            &fx.pool,
            "c1",
            &[("a1", "fetch"), ("a2", "fetch")],
            &[("a1", evicted), ("a2", "live".repeat(10))],
        )
        .await
        .expect("seed");

        let rows = store.tool_usage("c1").await.expect("aggregate");
        let fetch = by_name(&rows, "fetch");
        assert_eq!(
            fetch.call_count, 2,
            "eviction rewrites the RESULT, never the call record — the frequency \
             axis must be untouched by compaction"
        );
        assert_eq!(fetch.evicted_results, 1, "the evicted result is counted");
        assert_eq!(
            fetch.result_bytes, 40,
            "an evicted result contributes no resident bytes"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_usage_scoped_to_user_and_conversation() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        seed(&fx.pool, "c1", &[("a1", "search")], &[])
            .await
            .expect("seed u1/c1");
        seed(&fx.pool, "c2", &[("b1", "other")], &[])
            .await
            .expect("seed u1/c2");
        let rows = store.tool_usage("c1").await.expect("aggregate");
        assert_eq!(rows.len(), 1, "a sibling conversation must not leak in");
        assert_eq!(rows[0].tool_name, "search");
    })
    .await;
    // A different user sees nothing of u1's conversation — fail closed.
    with_user_id(UserId::from("u2"), async {
        let rows = store.tool_usage("c1").await.expect("aggregate");
        assert!(
            rows.is_empty(),
            "cross-user read must return nothing: {rows:?}"
        );
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_usage_empty_conversation() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        seed(&fx.pool, "c1", &[], &[]).await.expect("seed");
        let rows = store.tool_usage("c1").await.expect("must not error");
        assert!(rows.is_empty(), "no tool calls ⇒ empty vec, not an error");
    })
    .await;
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_usage_reports_first_and_last_ordinal() {
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skipping: TEST_DATABASE_URL unset");
        return;
    };
    let store = PgToolUsageStore::new(fx.pool.clone());
    with_user_id(UserId::from("u1"), async {
        seed(
            &fx.pool,
            "c1",
            &[("a1", "search"), ("b1", "fetch"), ("a2", "search")],
            &[],
        )
        .await
        .expect("seed");
        let rows = store.tool_usage("c1").await.expect("aggregate");
        let search = by_name(&rows, "search");
        assert!(
            search.first_ordinal < search.last_ordinal,
            "ordinals must bracket the calls: {search:?}"
        );
    })
    .await;
    fx.cleanup().await;
}
