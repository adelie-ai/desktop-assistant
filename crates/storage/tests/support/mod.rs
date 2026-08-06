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

use std::io;
use std::sync::Once;
use std::sync::{Arc, Mutex};

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

static SKIP_BANNER: Once = Once::new();

/// Provision the #434 RLS tool role for a DB-gated read-path suite,
/// simulating the privileged bootstrap (`crates/storage/bootstrap/rls_role.sql`)
/// that a superuser runs once in production.
///
/// Migration 029 (auto-run) is deliberately owner-only — it enables RLS and
/// creates the policies but does NOT create the `adele_query` role or grant it
/// anything, because the daemon's real DB role is un-privileged and cannot.
/// So the tests, which connect as a superuser, stand in for the DBA: create
/// the role, grant the connecting role membership so it can `SET LOCAL ROLE`,
/// and grant `USAGE` + `SELECT` on the suite's private schema (production
/// tables live in `public`; the private schema is a test-parallelism artifact).
///
/// Idempotent — the role is cluster-global and shared across suites.
pub async fn provision_tool_role(pool: &PgPool, schema: &str) {
    let role = desktop_assistant_storage::TOOL_QUERY_ROLE;
    // Create the restricted role once. This runs concurrently across the
    // suite's parallel tests against one cluster-global role, so a plain
    // `IF NOT EXISTS` check-then-create races; attempt the create and swallow
    // the duplicate (either variant Postgres may raise under the race).
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "DO $$ BEGIN \
           CREATE ROLE {role} NOLOGIN NOBYPASSRLS; \
         EXCEPTION WHEN duplicate_object OR unique_violation THEN NULL; \
         END $$;"
    )))
    .execute(pool)
    .await
    .expect("create tool role");
    for stmt in [
        // Membership so the (superuser) connecting role can SET LOCAL ROLE.
        format!("GRANT {role} TO CURRENT_USER WITH ADMIN OPTION"),
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

/// An `io::Write` sink that appends into a shared buffer, so a `fmt` layer's
/// writer closures (which each construct a fresh handle) all land in the same
/// place.
#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl io::Write for SharedBuf {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("lock capture buffer")
            .extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

static PERMISSIVE_GLOBAL_DEFAULT: Once = Once::new();

/// Install a permissive baseline subscriber as the process-wide global
/// default, once per test binary.
///
/// `tracing` caches each callsite's `Interest` globally, not per thread. Without
/// this, a callsite first evaluated on a thread that has no `set_default`
/// override (the ambient no-op default, which accepts nothing) latches
/// "never" for the whole process -- and then a *different* thread's
/// `capture_tracing` call below never sees that callsite's event at all,
/// because interest is checked before dispatch, upstream of which subscriber
/// is current. This showed up as a rare flake: the assertion failed only when
/// several tests using `capture_tracing` ran concurrently with tests that
/// don't, never when run alone.
///
/// The fix is to make every subscriber this module ever installs -- this one
/// and the per-call one below -- accept everything unconditionally, so no
/// concurrent cache rebuild can ever regress a callsite back to "never".
fn ensure_permissive_global_default() {
    PERMISSIVE_GLOBAL_DEFAULT.call_once(|| {
        let sink = tracing_subscriber::fmt().with_writer(io::sink).finish();
        tracing::subscriber::set_global_default(sink)
            .expect("install the permissive global default subscriber exactly once");
    });
}

/// Run `f` under a `fmt` subscriber that writes every emitted event into a
/// buffer, and return `f`'s result alongside the captured text. Used to
/// assert on a log line's content -- e.g. that a warning names both the
/// returned and the expected vector count -- rather than only on database
/// state, which cannot tell "cleared because of a count mismatch" apart from
/// "cleared because the backend errored".
///
/// Safe to hold across `.await`: `#[tokio::test]`'s default `current_thread`
/// flavor never migrates a task to another OS thread mid-poll, so the
/// thread-local default subscriber `set_default` installs stays in force for
/// `f`'s entire run, not just its first poll.
pub async fn capture_tracing<F, Fut, T>(f: F) -> (T, String)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    ensure_permissive_global_default();
    let buf = SharedBuf(Arc::new(Mutex::new(Vec::new())));
    let for_writer = buf.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || for_writer.clone())
        .with_ansi(false)
        .with_level(false)
        .with_target(false)
        .finish();
    let guard = tracing::subscriber::set_default(subscriber);
    let result = f().await;
    drop(guard);
    let bytes = buf.0.lock().expect("lock capture buffer").clone();
    (
        result,
        String::from_utf8(bytes).expect("captured log output is UTF-8"),
    )
}
