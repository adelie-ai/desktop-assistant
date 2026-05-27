//! Integration tests for the LLM-facing `execute_database_query` tool —
//! the security contract added in issue #141.
//!
//! The tool was originally a thin wrapper around an arbitrary
//! Postgres-text-to-result pipeline. The pre-#141 docstring claimed
//! "scratch isolation" but in practice
//!
//! - reads against `public.<personal-data-table>` returned rows for
//!   *every* user (no `user_id` filter ever applied), and
//! - qualified writes against `public.*` happily hit the production
//!   schema (the `search_path = scratch, public` only redirected
//!   *unqualified* writes).
//!
//! That negated the multi-tenant work in #102 / #105. This suite pins
//! the new contract:
//!
//! - SELECTs against personal-data tables are *transparently grafted*
//!   with `WHERE <table>.user_id = $N` so the caller only ever sees
//!   their own rows. An attempt to provide a different `user_id` value
//!   in the WHERE is overridden — the grafted predicate is AND'd in,
//!   so the intersection is empty.
//! - INSERT / UPDATE / DELETE / DROP / TRUNCATE / COPY / GRANT and the
//!   like are rejected at the parser level for any reference to a
//!   personal-data table (qualified or not), and for any compound
//!   statement.
//! - Unqualified writes against the `scratch` schema continue to work
//!   (the prior contract for ad-hoc relational work the LLM uses for
//!   intermediate joins, staging tables, materialized views, etc.).
//! - Reads against the Postgres system catalogs and the
//!   `tool_definitions` table (system-wide registry, see
//!   `audit_user_id_scoping.rs`) pass through without grafting.
//!
//! ## Running locally
//!
//! Set `TEST_DATABASE_URL` to a Postgres URL where the connecting role
//! can `CREATE SCHEMA` and `CREATE EXTENSION vector` (matches the
//! convention used by `user_id_scoping.rs` and
//! `multi_tenant_schema.rs`):
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage \
//!         --test database_query_user_id_scoping
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log
//! line so the suite stays green without a DB.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, Message, Role};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{
    PgConversationStore, UserId, execute_database_query, run_migrations, with_user_id,
};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// RAII fixture: private schema, pool pinned to it, migrations applied.
/// Mirrors the fixture in `user_id_scoping.rs` so the two suites stay
/// shape-aligned.
struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let schema = format!("issue141_{}", Uuid::now_v7().simple());

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

        // The pool pins `search_path` to the private schema, with
        // `public` retained so pgvector's `vector` type resolves. Note
        // that `public` here is the per-test `public` of the Postgres
        // database under test — *not* the production `public`. The
        // tests in this file specifically exercise the rule that the
        // tool must not reach across the `public` boundary, so we
        // construct each test inside its own throwaway schema and
        // *call* it `public.*` only inside the SQL we submit. The
        // schema-private "personal data" tables are created by
        // `run_migrations` directly inside the private schema, which
        // is exactly the production layout when search_path resolves
        // unqualified names.
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
            let _ = sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA \"{}\" CASCADE", self.schema)))
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

fn make_conversation(id: &str, title: &str, content: &str) -> Conversation {
    let mut conv = Conversation::new(id, title);
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, content));
    conv
}

/// Helper: seed alice + bob with one conversation each.
async fn seed_two_users(pool: &PgPool) {
    let store = PgConversationStore::new(pool.clone());
    with_user_id(UserId::new("alice"), async {
        store
            .create(make_conversation(
                "conv-alice",
                "alice's chat",
                "alice's secret message",
            ))
            .await
            .expect("alice create");
    })
    .await;
    with_user_id(UserId::new("bob"), async {
        store
            .create(make_conversation(
                "conv-bob",
                "bob's chat",
                "bob's secret message",
            ))
            .await
            .expect("bob create");
    })
    .await;
}

// ---------------------------------------------------------------------------
// Read-path tests — SELECT against personal-data tables MUST be scoped.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn select_with_user_id_filter_returns_only_caller_rows() {
    // Acceptance test from the issue brief: caller is alice; query
    // visits `conversations`; only alice's row(s) come back. This is
    // the *common case* — the LLM writes `SELECT id, title FROM
    // conversations`, the tool grafts `WHERE conversations.user_id =
    // $1` for alice, and bob's row is filtered out at the database
    // before any data leaves the pool.
    with_fixture(
        "select_with_user_id_filter_returns_only_caller_rows",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            let result = with_user_id(UserId::new("alice"), async {
                execute_database_query(
                    &fx.pool,
                    "SELECT id, title FROM conversations ORDER BY id",
                    100,
                )
                .await
            })
            .await
            .expect("query succeeds");

            let rows = result["rows"].as_array().expect("rows array");
            assert_eq!(rows.len(), 1, "alice must see exactly her one row");
            assert_eq!(rows[0][0], "conv-alice");
            assert_eq!(rows[0][1], "alice's chat");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn select_from_public_table_without_user_id_is_rejected() {
    // The issue brief offers two acceptable behaviours: (a) reject the
    // query outright, or (b) transparently graft the WHERE clause so
    // the result is the caller's rows only. We picked (b) — graft. So
    // the *behaviour* this test pins is: a SELECT against a personal-
    // data table that *does not* mention user_id must NOT return rows
    // belonging to other users. Bob runs the same query alice ran in
    // the previous test; he must see only his own row, never alice's.
    with_fixture(
        "select_from_public_table_without_user_id_is_rejected",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            let result = with_user_id(UserId::new("bob"), async {
                execute_database_query(
                    &fx.pool,
                    "SELECT id, title FROM conversations ORDER BY id",
                    100,
                )
                .await
            })
            .await
            .expect("query succeeds");

            let rows = result["rows"].as_array().expect("rows array");
            assert_eq!(rows.len(), 1, "bob must see exactly his one row");
            assert_eq!(rows[0][0], "conv-bob");
            assert!(
                !rows.iter().any(|r| r[0] == "conv-alice"),
                "bob must NEVER see alice's row, got {rows:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn cross_user_attempt_via_explicit_user_id_value_is_rejected() {
    // The hostile case: bob knows alice's user_id (it's a JWT sub,
    // visible in tokens) and tries to read her data by spelling it
    // out: `SELECT id FROM conversations WHERE user_id = 'alice'`.
    // The grafted predicate is `bob.user_id = $1` (= 'bob'), AND'd
    // with the user-supplied predicate. The intersection is empty:
    // bob's rows where user_id = 'alice', i.e. no rows.
    with_fixture(
        "cross_user_attempt_via_explicit_user_id_value_is_rejected",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            let result = with_user_id(UserId::new("bob"), async {
                execute_database_query(
                    &fx.pool,
                    "SELECT id, title FROM conversations WHERE user_id = 'alice'",
                    100,
                )
                .await
            })
            .await
            .expect("query succeeds (grafted predicate makes it empty)");

            let rows = result["rows"].as_array().expect("rows array");
            assert!(
                rows.is_empty(),
                "explicit cross-user WHERE must yield zero rows, got {rows:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn select_messages_does_not_leak_other_users_content() {
    // A second personal-data table (`messages`) — same contract.
    // Distinct from the previous test in that it exercises the
    // `content` column, which is what an exfiltration attempt would
    // actually target (the LLM has many ways to ask, but
    // `SELECT content FROM messages` is the obvious one).
    with_fixture(
        "select_messages_does_not_leak_other_users_content",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            let result = with_user_id(UserId::new("bob"), async {
                execute_database_query(
                    &fx.pool,
                    "SELECT content FROM messages ORDER BY id",
                    100,
                )
                .await
            })
            .await
            .expect("query succeeds");

            let rows = result["rows"].as_array().expect("rows array");
            for row in rows {
                let content = row[0].as_str().unwrap_or("");
                assert!(
                    !content.contains("alice"),
                    "bob's results must not contain alice's content: {content:?}"
                );
            }
            // Bob's row IS present.
            assert!(
                rows.iter()
                    .any(|r| r[0].as_str().unwrap_or("").contains("bob")),
                "bob should see his own message, got {rows:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn business_outcome_llm_cannot_exfiltrate_other_users_messages() {
    // End-to-end version of the previous test wired through the
    // `BuiltinToolService::execute_tool` path that the LLM actually
    // calls. Builds the closure exactly as `crates/daemon/src/main.rs`
    // wires it in production (see the `with_database` block around
    // line 919), then invokes the tool by name with a SELECT that an
    // exfiltration-minded LLM would write. Bob must see only his own
    // content.
    use desktop_assistant_core::ports::database::DbQueryFn;
    use desktop_assistant_mcp_client::executor::BuiltinToolService;

    with_fixture(
        "business_outcome_llm_cannot_exfiltrate_other_users_messages",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            let pool = fx.pool.clone();
            let query_fn: DbQueryFn = Arc::new(move |sql, limit| {
                let pool = pool.clone();
                Box::pin(async move {
                    execute_database_query(&pool, &sql, limit).await
                })
            });

            let service = BuiltinToolService::new().with_database(query_fn);

            // Bob's turn. The transport handler would install
            // `with_user_id(UserId::new("bob"), …)` around the LLM
            // tool-call dispatch; we do the same here.
            let response = with_user_id(UserId::new("bob"), async {
                service
                    .execute_tool(
                        "builtin_db_query",
                        serde_json::json!({
                            "query": "SELECT content FROM messages ORDER BY id"
                        }),
                    )
                    .await
            })
            .await
            .expect("tool succeeds");

            let json: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert_eq!(json["ok"], serde_json::json!(true));
            let rows = json["result"]["rows"].as_array().expect("rows array");
            for row in rows {
                let content = row[0].as_str().unwrap_or("");
                assert!(
                    !content.contains("alice"),
                    "LLM-driven SELECT for bob must not surface alice's content: {content:?}"
                );
            }
            assert!(
                rows.iter()
                    .any(|r| r[0].as_str().unwrap_or("").contains("bob")),
                "bob should see his own message via the tool, got {rows:?}"
            );
            fx
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Write-path tests — DDL/DML must not touch personal-data tables.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unqualified_drop_in_scratch_namespace_succeeds_but_does_not_touch_public() {
    // The prior contract (preserved): the LLM uses the `scratch`
    // schema for staging tables, intermediate joins, and other
    // ad-hoc relational work. An unqualified `CREATE TABLE foo` or
    // `DROP TABLE foo` resolves to `scratch.foo` because the write
    // path's transaction sets `search_path TO scratch, public`. The
    // production `public.conversations` row count must not change as
    // a side effect of any of these operations.
    with_fixture(
        "unqualified_drop_in_scratch_namespace_succeeds_but_does_not_touch_public",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            // Establish a staging table inside scratch, then drop it.
            // Both statements use unqualified names and should land in
            // scratch via search_path.
            let _ = execute_database_query(
                &fx.pool,
                "CREATE TABLE staging_foo (id INT PRIMARY KEY)",
                100,
            )
            .await
            .expect("create in scratch");

            let _ = execute_database_query(&fx.pool, "DROP TABLE staging_foo", 100)
                .await
                .expect("drop in scratch");

            // The production conversations table is unchanged.
            let count: (i64,) = sqlx::query_as("SELECT count(*) FROM conversations")
                .fetch_one(&fx.pool)
                .await
                .expect("count survives");
            assert_eq!(count.0, 2, "scratch DDL must not affect public tables");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn qualified_drop_against_public_is_rejected() {
    // The headline footgun: an LLM that knows the tool exists tries
    // `DROP TABLE public.conversations` to escape the search_path
    // redirect. Pre-#141 this would have succeeded. Now it must be
    // rejected at the parser level — *before* any SQL touches the
    // pool — and the table must still exist after.
    with_fixture(
        "qualified_drop_against_public_is_rejected",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            let result = execute_database_query(
                &fx.pool,
                "DROP TABLE public.conversations",
                100,
            )
            .await;
            assert!(
                matches!(result, Err(CoreError::ToolExecution(_))),
                "qualified DROP against public must be rejected, got {result:?}"
            );

            // The table is intact.
            let count: (i64,) = sqlx::query_as("SELECT count(*) FROM conversations")
                .fetch_one(&fx.pool)
                .await
                .expect("count after rejected drop");
            assert_eq!(count.0, 2, "rejected DROP must not have run");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn compound_statement_is_rejected() {
    // Statement-stuffing: `SELECT 1; DROP TABLE conversations` would
    // sneak the second statement past a single-statement classifier
    // that only looks at the first token. sqlparser parses both, so
    // we must reject anything that produced more than one statement.
    with_fixture("compound_statement_is_rejected", |fx| async move {
        seed_two_users(&fx.pool).await;

        let result = execute_database_query(
            &fx.pool,
            "SELECT 1; DROP TABLE conversations",
            100,
        )
        .await;
        assert!(
            matches!(result, Err(CoreError::ToolExecution(_))),
            "compound statement must be rejected, got {result:?}"
        );

        let count: (i64,) = sqlx::query_as("SELECT count(*) FROM conversations")
            .fetch_one(&fx.pool)
            .await
            .expect("count after rejected compound");
        assert_eq!(count.0, 2, "rejected compound must not have dropped the table");
        fx
    })
    .await;
}

#[tokio::test]
async fn non_select_statement_against_personal_data_is_rejected() {
    // An INSERT, UPDATE, or DELETE that *names* a personal-data
    // table must be rejected — regardless of qualification. The
    // read path is the only path the LLM may use to touch
    // personal-data tables; writes go to scratch (or to custom user
    // schemas), never to `public.*` personal data.
    with_fixture(
        "non_select_statement_against_personal_data_is_rejected",
        |fx| async move {
            seed_two_users(&fx.pool).await;

            // Unqualified INSERT — resolves to scratch via search_path
            // normally, but the parser flags it because `conversations`
            // is a reserved personal-data table name regardless of
            // schema.
            let r = execute_database_query(
                &fx.pool,
                "INSERT INTO conversations (id, title) VALUES ('x', 'y')",
                100,
            )
            .await;
            assert!(
                matches!(r, Err(CoreError::ToolExecution(_))),
                "unqualified INSERT into a personal-data table name must be rejected, got {r:?}"
            );

            // Qualified UPDATE — bypasses search_path; same outcome.
            let r = execute_database_query(
                &fx.pool,
                "UPDATE public.conversations SET title = 'hacked'",
                100,
            )
            .await;
            assert!(
                matches!(r, Err(CoreError::ToolExecution(_))),
                "qualified UPDATE against public.conversations must be rejected, got {r:?}"
            );

            // Qualified DELETE — same.
            let r = execute_database_query(
                &fx.pool,
                "DELETE FROM public.messages WHERE 1=1",
                100,
            )
            .await;
            assert!(
                matches!(r, Err(CoreError::ToolExecution(_))),
                "qualified DELETE against public.messages must be rejected, got {r:?}"
            );

            // None of those took effect.
            let count: (i64,) = sqlx::query_as("SELECT count(*) FROM conversations")
                .fetch_one(&fx.pool)
                .await
                .expect("count survives");
            assert_eq!(count.0, 2, "rejected writes must not have run");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn select_against_system_catalog_passes_through_ungrafted() {
    // Defense-in-depth doesn't require breaking legitimate uses:
    // `information_schema` and `pg_catalog` aren't personal-data
    // tables and have no `user_id` column. The grafter must skip
    // them. (If it tried to graft, the query would fail with an
    // "unknown column user_id" error.)
    with_fixture(
        "select_against_system_catalog_passes_through_ungrafted",
        |fx| async move {
            let result = execute_database_query(
                &fx.pool,
                "SELECT table_name FROM information_schema.tables \
                 WHERE table_schema = 'pg_catalog' ORDER BY table_name LIMIT 1",
                100,
            )
            .await
            .expect("system-catalog read should succeed");

            let rows = result["rows"].as_array().expect("rows array");
            assert!(!rows.is_empty(), "expected at least one row from pg_catalog");
            fx
        },
    )
    .await;
}
