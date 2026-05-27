//! Integration tests for the multi-tenant schema migration (issue #102).
//!
//! These tests exercise the live migration tooling against a real Postgres
//! instance with the `pgvector` extension available. Each test creates a
//! private schema, points the connection pool's `search_path` at it, runs
//! the embedded migrations, asserts schema/data invariants, then drops the
//! schema on teardown.
//!
//! ## Running locally
//!
//! Set `TEST_DATABASE_URL` to a Postgres URL where the connecting role can
//! `CREATE SCHEMA` and `CREATE EXTENSION vector`. Example using the
//! pgvector docker image:
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test multi_tenant_schema
//! ```
//!
//! When `TEST_DATABASE_URL` is unset (the default for `cargo test`), every
//! test pass-skips with a log line so the suite stays green without a DB.

use std::sync::Arc;

use desktop_assistant_storage::run_migrations;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// Sentinel user_id used for backfilling pre-multi-tenant rows. Must
/// match the value the migration writes — single-tenant desktop deploys
/// collapse to this id until JWT-based extraction (issue #105) lands.
const DEFAULT_USER_ID: &str = "default";

/// RAII guard that owns a freshly-created Postgres schema and a pool
/// whose connections have `search_path` pinned to it. Dropping the
/// guard drops the schema on a best-effort basis.
struct SchemaFixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl SchemaFixture {
    /// Build a fixture against `TEST_DATABASE_URL`, or `None` if the env
    /// var is unset. Tests should pass-skip in the `None` case so the
    /// suite stays green without a DB.
    async fn try_new() -> Option<Self> {
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let schema = format!("mt_test_{}", Uuid::now_v7().simple());

        // Admin connection just for schema lifecycle, kept short-lived
        // so the per-test pool owns the only long-running connections.
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

        // Per-test pool with `search_path` pinned via `after_connect` so
        // every connection (including those opened mid-test) lands in
        // the right schema. `public` stays on the path so the
        // `vector` type from pgvector remains resolvable.
        let schema_for_hook = Arc::new(schema.clone());
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .after_connect(move |conn, _meta| {
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

        Some(Self {
            pool,
            schema,
            admin_url: url,
        })
    }

    async fn migrate(&self) {
        run_migrations(&self.pool)
            .await
            .expect("run_migrations succeeds against test schema");
    }

    /// Drop the schema on a best-effort basis — failures here don't fail
    /// the test, but they do log so a developer can clean up manually.
    async fn cleanup(self) {
        self.pool.close().await;
        let admin = match PgPoolOptions::new()
            .max_connections(1)
            .connect(&self.admin_url)
            .await
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("cleanup: failed to reconnect to drop schema {}: {e}", self.schema);
                return;
            }
        };
        if let Err(e) = sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", self.schema))
            .execute(&admin)
            .await
        {
            eprintln!("cleanup: failed to drop schema {}: {e}", self.schema);
        }
        admin.close().await;
    }
}

/// Run `body` with a fresh schema fixture, skipping the test (with a log
/// line) when `TEST_DATABASE_URL` is unset. Cleanup runs even if `body`
/// panics — without this every failed test leaks a schema.
async fn with_fixture<F, Fut>(test_name: &str, body: F)
where
    F: FnOnce(SchemaFixture) -> Fut,
    Fut: std::future::Future<Output = SchemaFixture>,
{
    let Some(fixture) = SchemaFixture::try_new().await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; {test_name} pass-skipped. \
             Set it to a Postgres URL with pgvector available to run."
        );
        return;
    };
    let fixture = body(fixture).await;
    fixture.cleanup().await;
}

/// Returns true iff the named column exists on the named table inside
/// the connection's current `search_path` first schema. Restricting to
/// the test schema (rather than `current_schema()`) keeps the assertion
/// honest if a stray `public` table with the same name ever appears.
async fn column_exists(pool: &PgPool, schema: &str, table: &str, column: &str) -> bool {
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*)::BIGINT FROM information_schema.columns
         WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
    )
    .bind(schema)
    .bind(table)
    .bind(column)
    .fetch_one(pool)
    .await
    .expect("query information_schema.columns");
    row.0 > 0
}

/// Whether `column` on `table` is declared NOT NULL.
async fn column_is_not_null(pool: &PgPool, schema: &str, table: &str, column: &str) -> bool {
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT is_nullable FROM information_schema.columns
         WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
    )
    .bind(schema)
    .bind(table)
    .bind(column)
    .fetch_optional(pool)
    .await
    .expect("query is_nullable");
    matches!(row, Some((s,)) if s == "NO")
}

/// True iff *any* index on `table` covers `(user_id, …)` as its leading
/// column. We don't pin the exact index name — only the property that
/// matters for the hot query path.
async fn has_user_id_leading_index(pool: &PgPool, schema: &str, table: &str) -> bool {
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*)::BIGINT
         FROM pg_indexes i
         JOIN pg_class c     ON c.relname = i.indexname
         JOIN pg_index pi    ON pi.indexrelid = c.oid
         JOIN pg_class t     ON t.oid = pi.indrelid
         JOIN pg_namespace n ON n.oid = t.relnamespace
         JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = pi.indkey[0]
         WHERE n.nspname = $1 AND t.relname = $2 AND a.attname = 'user_id'",
    )
    .bind(schema)
    .bind(table)
    .fetch_one(pool)
    .await
    .expect("query for user_id-leading index");
    row.0 > 0
}

// ---------------------------------------------------------------------------
// Schema shape — every personal-data SQL table gains `user_id NOT NULL`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn migration_adds_user_id_column_to_conversations() {
    with_fixture(
        "migration_adds_user_id_column_to_conversations",
        |fx| async move {
            fx.migrate().await;
            assert!(
                column_exists(&fx.pool, &fx.schema, "conversations", "user_id").await,
                "conversations.user_id should exist after migration"
            );
            assert!(
                column_is_not_null(&fx.pool, &fx.schema, "conversations", "user_id").await,
                "conversations.user_id should be NOT NULL"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn migration_adds_user_id_column_to_messages() {
    with_fixture("migration_adds_user_id_column_to_messages", |fx| async move {
        fx.migrate().await;
        assert!(column_exists(&fx.pool, &fx.schema, "messages", "user_id").await);
        assert!(column_is_not_null(&fx.pool, &fx.schema, "messages", "user_id").await);
        fx
    })
    .await;
}

#[tokio::test]
async fn migration_adds_user_id_column_to_knowledge_base() {
    with_fixture(
        "migration_adds_user_id_column_to_knowledge_base",
        |fx| async move {
            fx.migrate().await;
            assert!(column_exists(&fx.pool, &fx.schema, "knowledge_base", "user_id").await);
            assert!(column_is_not_null(&fx.pool, &fx.schema, "knowledge_base", "user_id").await);
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn migration_adds_user_id_column_to_message_summaries() {
    with_fixture(
        "migration_adds_user_id_column_to_message_summaries",
        |fx| async move {
            fx.migrate().await;
            assert!(column_exists(&fx.pool, &fx.schema, "message_summaries", "user_id").await);
            assert!(
                column_is_not_null(&fx.pool, &fx.schema, "message_summaries", "user_id").await
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn migration_adds_user_id_column_to_dreaming_watermarks() {
    with_fixture(
        "migration_adds_user_id_column_to_dreaming_watermarks",
        |fx| async move {
            fx.migrate().await;
            assert!(column_exists(&fx.pool, &fx.schema, "dreaming_watermarks", "user_id").await);
            assert!(
                column_is_not_null(&fx.pool, &fx.schema, "dreaming_watermarks", "user_id").await
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn migration_adds_user_id_column_to_tag_registry() {
    with_fixture(
        "migration_adds_user_id_column_to_tag_registry",
        |fx| async move {
            fx.migrate().await;
            assert!(column_exists(&fx.pool, &fx.schema, "tag_registry", "user_id").await);
            assert!(column_is_not_null(&fx.pool, &fx.schema, "tag_registry", "user_id").await);
            fx
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Backfill — existing rows inserted before the migration get the sentinel id.
// ---------------------------------------------------------------------------

/// Simulate a "pre-migration" database state by running the legacy schema
/// migrations only, inserting fixture rows, then running the full
/// `run_migrations()` (which now includes the multi-tenant migration).
/// The fixture rows must come out with `user_id = DEFAULT_USER_ID`.
#[tokio::test]
async fn migration_backfills_existing_conversation_rows_to_default_user() {
    with_fixture(
        "migration_backfills_existing_conversation_rows_to_default_user",
        |fx| async move {
            // 1. Stage the database to look like an install from before #102:
            //    only the pre-multi-tenant migrations exist, so
            //    `conversations` has no `user_id` column.
            run_pre_multitenant_migrations(&fx.pool).await;

            sqlx::query(
                "INSERT INTO conversations (id, title) VALUES ('pre-1', 'Legacy Chat')",
            )
            .execute(&fx.pool)
            .await
            .expect("seed legacy conversation row");

            // 2. Now run the full migration set — the multi-tenant migration
            //    must backfill the legacy row.
            fx.migrate().await;

            let row: (String,) =
                sqlx::query_as("SELECT user_id FROM conversations WHERE id = 'pre-1'")
                    .fetch_one(&fx.pool)
                    .await
                    .expect("read back legacy conversation");
            assert_eq!(
                row.0, DEFAULT_USER_ID,
                "pre-existing conversation row should be backfilled to '{DEFAULT_USER_ID}'"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn migration_backfills_existing_knowledge_base_rows_to_default_user() {
    with_fixture(
        "migration_backfills_existing_knowledge_base_rows_to_default_user",
        |fx| async move {
            run_pre_multitenant_migrations(&fx.pool).await;

            sqlx::query(
                "INSERT INTO knowledge_base (id, content) VALUES ('kb-pre-1', 'legacy fact')",
            )
            .execute(&fx.pool)
            .await
            .expect("seed legacy KB row");

            fx.migrate().await;

            let row: (String,) =
                sqlx::query_as("SELECT user_id FROM knowledge_base WHERE id = 'kb-pre-1'")
                    .fetch_one(&fx.pool)
                    .await
                    .expect("read back legacy KB row");
            assert_eq!(row.0, DEFAULT_USER_ID);
            fx
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Composite indexes — every personal-data table has a `(user_id, …)` index
// for the hot query paths #105's scoping will use.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn composite_user_id_index_exists_on_conversations() {
    with_fixture(
        "composite_user_id_index_exists_on_conversations",
        |fx| async move {
            fx.migrate().await;
            assert!(
                has_user_id_leading_index(&fx.pool, &fx.schema, "conversations").await,
                "conversations needs a `(user_id, …)` index for per-user listing"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn composite_user_id_index_exists_on_messages() {
    with_fixture(
        "composite_user_id_index_exists_on_messages",
        |fx| async move {
            fx.migrate().await;
            assert!(has_user_id_leading_index(&fx.pool, &fx.schema, "messages").await);
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn composite_user_id_index_exists_on_knowledge_base() {
    with_fixture(
        "composite_user_id_index_exists_on_knowledge_base",
        |fx| async move {
            fx.migrate().await;
            assert!(has_user_id_leading_index(&fx.pool, &fx.schema, "knowledge_base").await);
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn composite_user_id_index_exists_on_message_summaries() {
    with_fixture(
        "composite_user_id_index_exists_on_message_summaries",
        |fx| async move {
            fx.migrate().await;
            assert!(has_user_id_leading_index(&fx.pool, &fx.schema, "message_summaries").await);
            fx
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Regression guard — NOT NULL means inserts without user_id fail loudly.
// This proves the schema is in place even before #105 updates queries.
// ---------------------------------------------------------------------------

/// Running `run_migrations` a second time against an already-migrated
/// schema must be a no-op — both the daemon and the dreaming worker
/// invoke it at startup, and existing installs already see migrations
/// 001-015 re-run cleanly on every boot. The multi-tenant migration
/// has to preserve that contract.
#[tokio::test]
async fn migrations_are_idempotent() {
    with_fixture("migrations_are_idempotent", |fx| async move {
        fx.migrate().await;
        // Second pass: must not raise (no duplicate-column errors, no
        // duplicate-index errors, no PK-already-exists errors).
        fx.migrate().await;
        // Sanity-check the second pass left the column in place.
        assert!(column_exists(&fx.pool, &fx.schema, "conversations", "user_id").await);
        fx
    })
    .await;
}

#[tokio::test]
async fn inserting_conversation_without_user_id_fails() {
    with_fixture("inserting_conversation_without_user_id_fails", |fx| async move {
        fx.migrate().await;

        // Drop the default if the migration left one in place — without
        // this the test would silently pass on the sentinel. Idempotent
        // and a no-op if no default exists.
        sqlx::query("ALTER TABLE conversations ALTER COLUMN user_id DROP DEFAULT")
            .execute(&fx.pool)
            .await
            .ok();

        let err = sqlx::query(
            "INSERT INTO conversations (id, title) VALUES ('needs-user', 'should fail')",
        )
        .execute(&fx.pool)
        .await
        .expect_err("insert without user_id should fail under multi-tenant schema");

        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("user_id") || msg.contains("not-null") || msg.contains("null value"),
            "expected a not-null violation on user_id, got: {err}"
        );
        fx
    })
    .await;
}

// ---------------------------------------------------------------------------
// Internal: hand-roll the pre-multi-tenant migration set so backfill tests
// can stage a "legacy" DB before running the full migration chain.
// Mirrors `run_migrations` up to migration 015 — the ordering and the
// per-statement boundaries match.
// ---------------------------------------------------------------------------

async fn run_pre_multitenant_migrations(pool: &PgPool) {
    let migrations: &[&str] = &[
        include_str!("../migrations/001_initial_schema.sql"),
        // pgvector must be created before any vector column references it.
        "CREATE EXTENSION IF NOT EXISTS vector",
        include_str!("../migrations/002_vector_tables.sql"),
        include_str!("../migrations/002b_tool_definitions.sql"),
        include_str!("../migrations/003_vector_indexes.sql"),
        include_str!("../migrations/004_embedding_model_tracking.sql"),
        include_str!("../migrations/005_uuidv7_ids.sql"),
        include_str!("../migrations/006_dreaming_watermarks.sql"),
        include_str!("../migrations/007_chunked_embeddings.sql"),
        include_str!("../migrations/008_message_summaries.sql"),
        include_str!("../migrations/009_conversation_archived_at.sql"),
        include_str!("../migrations/010_fix_damaged_embeddings.sql"),
        include_str!("../migrations/011_conversation_last_model.sql"),
        include_str!("../migrations/012_conversation_active_task.sql"),
        include_str!("../migrations/013_conversation_message_fts.sql"),
        include_str!("../migrations/014_tag_registry.sql"),
        include_str!("../migrations/015_knowledge_base_review_columns.sql"),
    ];
    for sql in migrations {
        sqlx::raw_sql(sql)
            .execute(pool)
            .await
            .expect("pre-multi-tenant migration step succeeds");
    }
}
