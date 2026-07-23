//! SQLite connection-pool construction and idempotent schema application.
//!
//! Mirrors the Postgres adapter's `pool.rs`: migrations are a hand-registered
//! list of `include_str!`'d, idempotent DDL scripts applied at pool init, not
//! auto-discovered from the directory (see the `every_migration_is_registered`
//! guard below).
//!
//! Every connection sets `PRAGMA foreign_keys = ON` so the `ON DELETE CASCADE`
//! / `ON DELETE SET NULL` actions in the schema are actually enforced — SQLite
//! leaves foreign-key enforcement off by default, per connection.

use std::str::FromStr;

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

/// Open a connection pool to a SQLite database at `url`, creating the file if
/// it does not exist, and apply the relational schema.
///
/// `url` is a standard sqlx SQLite URL (`sqlite://path/to/db.sqlite` or
/// `sqlite::memory:`). Foreign-key enforcement is enabled on every connection.
///
/// Note: an in-memory database (`sqlite::memory:`) is private to each
/// connection, so pass `max_connections = 1` for one (or use
/// [`create_memory_pool`], which pins a single persistent connection).
pub async fn create_pool(url: &str, max_connections: u32) -> Result<SqlitePool, sqlx::Error> {
    // TODO(sqlite inc1b): when this pool is actually wired for a file-backed
    // DB, enable WAL journal mode + a busy_timeout so concurrent readers don't
    // block on a writer. Left off here because inc1 is unwired and its tests use
    // a single-connection in-memory pool where WAL doesn't apply.
    let opts = SqliteConnectOptions::from_str(url)?
        .create_if_missing(true)
        .foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(max_connections)
        .connect_with(opts)
        .await?;
    run_migrations(&pool).await?;
    Ok(pool)
}

/// Open an ephemeral in-memory SQLite pool with the relational schema applied.
///
/// An in-memory database lives only as long as the connection that created it,
/// so the pool is pinned to exactly one connection that is never idle-reaped
/// or lifetime-expired. Intended for tests and ephemeral (throw-away) use.
pub async fn create_memory_pool() -> Result<SqlitePool, sqlx::Error> {
    let opts = SqliteConnectOptions::from_str("sqlite::memory:")?.foreign_keys(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect_with(opts)
        .await?;
    run_migrations(&pool).await?;
    Ok(pool)
}

/// Apply the embedded, idempotent DDL scripts to `pool`.
///
/// Increment 1 ships a single consolidated relational-backbone script. Later
/// increments append ordinally-numbered scripts (vector/FTS in inc2, etc.),
/// each registered here with its own `sqlx::raw_sql(include_str!(...))` call.
pub async fn run_migrations(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::raw_sql(include_str!("../migrations/001_relational_schema.sql"))
        .execute(pool)
        .await?;
    // #287: additive upgrade for dev DBs created before these columns landed in
    // the baseline (the CREATE TABLE IF NOT EXISTS above is a no-op against an
    // existing table). SQLite has no ADD COLUMN IF NOT EXISTS and this runner
    // re-executes every boot, so each ALTER is guarded on PRAGMA table_info to
    // stay idempotent. A fresh DB already has the columns from 001, so these are
    // no-ops there.
    ensure_column(
        pool,
        "background_tasks",
        "owner_todo",
        "TEXT NOT NULL DEFAULT ''",
    )
    .await?;
    ensure_column(pool, "background_tasks", "spawn_marker", "TEXT").await?;
    // Skill index (#594): relational + FTS5 (the repo's first FTS5 table).
    sqlx::raw_sql(include_str!("../migrations/002_skill_index.sql"))
        .execute(pool)
        .await?;
    // #639: presence columns for the cumulative catalog. A fresh DB gets them
    // from 002 above; these ALTERs upgrade a dev DB created before they landed.
    ensure_column(
        pool,
        "skill_index",
        "present_on_disk",
        "INTEGER NOT NULL DEFAULT 1",
    )
    .await?;
    ensure_column(pool, "skill_index", "last_seen_at", "TEXT").await?;
    Ok(())
}

/// Add `column` (declared as `type_decl`) to `table` if it is not already
/// present. Idempotent: safe to call on every startup. SQLite lacks
/// `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so this checks `PRAGMA
/// table_info` first. `table`/`column`/`type_decl` are internal string
/// constants (never external input), so the interpolation is injection-safe.
async fn ensure_column(
    pool: &SqlitePool,
    table: &str,
    column: &str,
    type_decl: &str,
) -> Result<(), sqlx::Error> {
    let cols: Vec<(String,)> = sqlx::query_as(sqlx::AssertSqlSafe(format!(
        "SELECT name FROM pragma_table_info('{table}')"
    )))
    .fetch_all(pool)
    .await?;
    if cols.iter().any(|(name,)| name == column) {
        return Ok(());
    }
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "ALTER TABLE {table} ADD COLUMN {column} {type_decl}"
    )))
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Every `.sql` file in `migrations/` must be wired into `run_migrations`.
    /// Migrations are a hand-maintained `include_str!` list, NOT auto-discovered
    /// — so an unregistered new file would compile fine and silently never run,
    /// surfacing only as a runtime "no such table" error. This guard turns that
    /// into a build-time failure instead (mirrors the Postgres adapter's guard).
    #[test]
    fn every_migration_is_registered() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");
        let source = include_str!("pool.rs");

        let mut unregistered: Vec<String> = std::fs::read_dir(dir)
            .expect("read migrations/ dir")
            .map(|e| e.expect("dir entry").file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".sql"))
            .filter(|name| !source.contains(name.as_str()))
            .collect();
        unregistered.sort();

        assert!(
            unregistered.is_empty(),
            "migration file(s) exist in migrations/ but are not referenced in \
             run_migrations() in pool.rs: {unregistered:?}"
        );
    }
}
