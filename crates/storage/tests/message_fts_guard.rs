//! Integration test for the message FTS INSERT guard (issue #177).
//!
//! Migration 013 added a generated `tsv` column over the full message
//! content. A multi-megabyte / high-entropy message could exceed Postgres's
//! hard 1 MB tsvector limit and abort the INSERT. Migration 021 stops
//! FTS-indexing `tool`-role rows and bounds the indexed input for other roles,
//! so a large message can always be stored while normal full-text search keeps
//! working.
//!
//! ## Running locally
//!
//! Set `TEST_DATABASE_URL` to a Postgres URL whose role can `CREATE SCHEMA`
//! (the `vector` extension must already be available in the target database):
//!
//! ```sh
//! TEST_DATABASE_URL="postgres://user:pw@localhost/db" \
//!     cargo test -p desktop-assistant-storage --test message_fts_guard
//! ```
//!
//! When `TEST_DATABASE_URL` is unset the test pass-skips with a log line.

mod support;

use std::sync::Arc;

use desktop_assistant_storage::run_migrations;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// RAII-ish fixture owning a freshly-created schema and a pool whose
/// connections have `search_path` pinned to it. `cleanup` drops the schema.
struct SchemaFixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl SchemaFixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("fts_test_{}", Uuid::now_v7().simple());

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
            .after_connect(move |conn, _meta| {
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

#[tokio::test]
async fn large_message_insert_succeeds_and_fts_still_works() {
    let Some(fixture) = SchemaFixture::try_new().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; large_message_insert_succeeds_and_fts_still_works \
             pass-skipped"
        );
        return;
    };

    run_migrations(&fixture.pool)
        .await
        .expect("migrations (incl. 021) apply against the test schema");

    // A parent conversation (all NOT NULL columns supplied).
    sqlx::query(
        "INSERT INTO conversations \
            (id, user_id, title, created_at, updated_at, context_summary, compacted_through) \
         VALUES ('c1', 'default', 't', now(), now(), '', 0)",
    )
    .execute(&fixture.pool)
    .await
    .expect("insert conversation");

    // ~4 MB of all-distinct tokens — under the OLD definition this produces a
    // >1 MB tsvector and the INSERT errors with "string is too long for
    // tsvector". Built in SQL so we don't ship megabytes from the test binary.
    let big_tool = sqlx::query(
        "INSERT INTO messages (id, conversation_id, user_id, ordinal, role, content) \
         SELECT 'm1', 'c1', 'default', 1, 'tool', string_agg('tok' || g, ' ') \
         FROM generate_series(1, 300000) g",
    )
    .execute(&fixture.pool)
    .await;
    assert!(
        big_tool.is_ok(),
        "large tool-result INSERT must succeed post-021, got {big_tool:?}"
    );

    // The same oversized payload as a non-tool role must also store (the
    // indexed input is bounded to 256 KiB so the tsvector stays under 1 MB).
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, user_id, ordinal, role, content) \
         SELECT 'm2', 'c1', 'default', 2, 'user', string_agg('tok' || g, ' ') \
         FROM generate_series(1, 300000) g",
    )
    .execute(&fixture.pool)
    .await
    .expect("large user-message INSERT must succeed post-021");

    // A normal message must remain full-text searchable.
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, user_id, ordinal, role, content) \
         VALUES ('m3', 'c1', 'default', 3, 'user', 'the quick brown fox')",
    )
    .execute(&fixture.pool)
    .await
    .expect("insert normal message");

    let tool_not_indexed: bool =
        sqlx::query_scalar("SELECT tsv = ''::tsvector FROM messages WHERE role = 'tool'")
            .fetch_one(&fixture.pool)
            .await
            .expect("read tool tsv");
    assert!(tool_not_indexed, "tool-role rows must not be FTS-indexed");

    let fts_hits: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM messages WHERE tsv @@ to_tsquery('english', 'fox')",
    )
    .fetch_one(&fixture.pool)
    .await
    .expect("run FTS query");
    assert_eq!(fts_hits, 1, "FTS must still match normal message content");

    fixture.cleanup().await;
}
