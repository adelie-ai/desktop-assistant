#![allow(dead_code)]
//! Shared test support for the DB-gated storage suites.
//!
//! Every isolation suite is gated on `TEST_DATABASE_URL` and pass-skips when
//! it is unset — but the skip used to be a single easy-to-miss `eprintln!`
//! line buried among passing tests, so a green `cargo test` run read as
//! "multi-tenant isolation: covered" when in fact it had verified nothing
//! (this is how the #431 cross-tenant db_query bug went unnoticed). This
//! module centralizes the gate so the skip is *loud* and actionable, and
//! points at the one-command self-provisioning harness (`just test-db`).
//!
//! Included by each integration test via `mod support;` (it lives in a
//! subdirectory so cargo does not compile it as its own test binary).

use std::sync::Arc;
use std::sync::Once;

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

static SKIP_BANNER: Once = Once::new();

/// Grant the #434 RLS tool role (`adele_query`, created by migration 029)
/// `USAGE` + `SELECT` on `schema`, so a read-path suite whose tables live
/// in a private test schema can `SET LOCAL ROLE adele_query` and still
/// resolve them. Production tables live in `public`, which migration 029
/// grants directly; this mirrors that grant for the private-schema layout
/// the DB-gated suites use for parallel isolation. Without it every grafted
/// SELECT under the tool role would fail with "permission denied".
pub async fn grant_tool_role_on_schema(pool: &PgPool, schema: &str) {
    let role = desktop_assistant_storage::TOOL_QUERY_ROLE;
    for stmt in [
        format!("GRANT USAGE ON SCHEMA \"{schema}\" TO {role}"),
        format!("GRANT SELECT ON ALL TABLES IN SCHEMA \"{schema}\" TO {role}"),
    ] {
        sqlx::query(sqlx::AssertSqlSafe(stmt))
            .execute(pool)
            .await
            .expect("grant tool role on test schema");
    }
}

/// The connection URL for the DB-gated suites, or `None` when no database is
/// available (in which case the caller should pass-skip). On the first `None`
/// in a test binary, prints a prominent, actionable banner so the skip is
/// impossible to mistake for "isolation is covered".
///
/// Set `TEST_DATABASE_URL` yourself, or run `just test-db` which boots an
/// ephemeral pgvector container (with the `vector` extension pre-created via
/// an auto-loaded init fixture), points this at it, runs the suites, and
/// tears the container down.
pub fn test_database_url() -> Option<String> {
    match std::env::var("TEST_DATABASE_URL") {
        Ok(url) if !url.trim().is_empty() => Some(url),
        _ => {
            SKIP_BANNER.call_once(print_skip_banner);
            None
        }
    }
}

fn print_skip_banner() {
    let banner = "\n\
         ┌──────────────────────────────────────────────────────────────────────┐\n\
         │  ⚠  storage DB-gated tests SKIPPED — TEST_DATABASE_URL is not set.     │\n\
         │                                                                        │\n\
         │  These verify multi-tenant user_id isolation. A green run WITHOUT a    │\n\
         │  database proves nothing about cross-tenant safety — it only means     │\n\
         │  the suites were skipped.                                              │\n\
         │                                                                        │\n\
         │  Run them against an ephemeral Postgres with:   just test-db           │\n\
         └──────────────────────────────────────────────────────────────────────┘\n";
    // libtest captures stdout/stderr for *passing* tests, so a plain
    // `eprintln!` here is hidden under a normal `cargo test` run (which is why
    // the old one-line skip was effectively invisible). Writing to the
    // controlling terminal bypasses that capture so the warning is actually
    // seen; fall back to stderr when there is no tty (CI, piped output — where
    // the `just test`/`just check` recipe-level warning covers it instead).
    use std::io::Write;
    match std::fs::OpenOptions::new().write(true).open("/dev/tty") {
        Ok(mut tty) => {
            let _ = tty.write_all(banner.as_bytes());
        }
        Err(_) => eprintln!("{banner}"),
    }
}

/// RAII fixture for the DB-touching dreaming / embedding suites: a freshly
/// created private schema, a pool whose connections pin `search_path` to it,
/// and all migrations applied. Dropping the schema is done explicitly via
/// [`DbFixture::cleanup`] so a panicking test still tears down.
///
/// `public` stays on the search path so the pgvector `vector` type (created
/// there by the test harness) remains resolvable inside the private schema.
pub struct DbFixture {
    pub pool: PgPool,
    schema: String,
    admin_url: String,
}

impl DbFixture {
    /// The private schema this fixture's tables live in — for suites that
    /// need to name it in a catalog query or a schema-scoped grant.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Build a fixture against `TEST_DATABASE_URL`, or `None` when it is unset
    /// (callers pass-skip). `prefix` disambiguates schemas across suites so a
    /// leaked schema is traceable to the suite that made it.
    pub async fn try_new(prefix: &str) -> Option<Self> {
        let url = test_database_url()?;
        let schema = format!("{prefix}_{}", Uuid::now_v7().simple());

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

        desktop_assistant_storage::run_migrations(&pool)
            .await
            .expect("run_migrations succeeds against test schema");

        Some(Self {
            pool,
            schema,
            admin_url: url,
        })
    }

    /// Drop the schema on a best-effort basis; failures log but don't fail the
    /// test (they'd only mask the real assertion).
    pub async fn cleanup(self) {
        self.pool.close().await;
        let admin = match PgPoolOptions::new()
            .max_connections(1)
            .connect(&self.admin_url)
            .await
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "cleanup: failed to reconnect to drop schema {}: {e}",
                    self.schema
                );
                return;
            }
        };
        if let Err(e) = sqlx::query(sqlx::AssertSqlSafe(format!(
            "DROP SCHEMA \"{}\" CASCADE",
            self.schema
        )))
        .execute(&admin)
        .await
        {
            eprintln!("cleanup: failed to drop schema {}: {e}", self.schema);
        }
        admin.close().await;
    }
}
