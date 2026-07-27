//! Integration tests for the migration version ledger (issue #730).
//!
//! `run_migrations` used to re-execute every `.sql` file on every daemon
//! boot. Migration 021 drops and re-adds `messages.tsv`, a STORED generated
//! column, so every boot rewrote the whole `messages` heap under ACCESS
//! EXCLUSIVE, rebuilt its GIN index, and burned one of the table's 1600
//! `pg_attribute` slots — startup cost proportional to history, and a hard
//! ceiling after ~1,588 boots.
//!
//! These suites pin the contract that replaces it: a `schema_migrations`
//! ledger so each file is applied at most once, and a `pg_advisory_lock` so
//! two daemons booting against one database serialize instead of racing.
//!
//! Acceptance criteria, each a named test below:
//! - `every_migration_is_recorded_in_the_version_table`
//! - `second_boot_does_not_drop_and_re_add_the_messages_tsv_column`
//! - `legacy_database_without_a_version_table_is_backfilled`
//! - `legacy_transition_applies_migrations_that_never_ran`
//! - `empty_version_table_on_a_migrated_database_replays_cleanly`
//! - `unrecognized_recorded_migration_is_ignored`
//! - `concurrent_boots_apply_each_migration_once`
//!
//! The ledger holds no per-user data; that it must stay out of the
//! multi-tenant boundary is pinned by `GLOBAL_TABLES` in
//! `database_query_user_id_scoping.rs`.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`).

mod support;

use std::sync::Arc;

use desktop_assistant_storage::run_migrations;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use support::DbFixture;

/// Names of the migration files on disk — the source of truth the ledger is
/// compared against.
fn migration_files_on_disk() -> Vec<String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read migrations/ dir")
        .map(|e| e.expect("dir entry").file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".sql"))
        .collect();
    names.sort();
    names
}

/// Migration names recorded in the ledger, sorted.
async fn recorded_migrations(pool: &PgPool) -> Vec<String> {
    let mut names: Vec<String> = sqlx::query_scalar("SELECT name FROM schema_migrations")
        .fetch_all(pool)
        .await
        .expect("read schema_migrations ledger");
    names.sort();
    names
}

/// Every `pg_attribute` slot the table has ever consumed, dropped columns
/// included. A drop/re-add of a column leaves the dropped entry behind and
/// allocates a fresh one, so this number is the direct probe for "did a
/// migration rewrite the table again?".
async fn attribute_slots(pool: &PgPool, schema: &str, table: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT count(*) FROM pg_attribute a \
         JOIN pg_class c     ON c.oid = a.attrelid \
         JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE n.nspname = $1 AND c.relname = $2 AND a.attnum > 0",
    )
    .bind(schema)
    .bind(table)
    .fetch_one(pool)
    .await
    .expect("count pg_attribute slots")
}

/// True when `table` has a live (not dropped) column named `column`.
async fn has_column(pool: &PgPool, schema: &str, table: &str, column: &str) -> bool {
    sqlx::query_scalar::<_, i64>(
        "SELECT count(*) FROM information_schema.columns \
         WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
    )
    .bind(schema)
    .bind(table)
    .bind(column)
    .fetch_one(pool)
    .await
    .expect("query information_schema for column")
        > 0
}

/// Put the schema into the state every pre-#730 database is in: fully
/// migrated (every file has been applied, because the old runner replayed
/// them all on every boot) but with no ledger to say so.
async fn simulate_pre_ledger_database(pool: &PgPool) {
    sqlx::query("DROP TABLE IF EXISTS schema_migrations")
        .execute(pool)
        .await
        .expect("drop the ledger to simulate a pre-#730 database");
}

/// A second pool onto the same private schema — a stand-in for a second
/// daemon booting against one database.
async fn second_pool(schema: &str) -> PgPool {
    let url = support::test_database_url().expect("caller checked TEST_DATABASE_URL");
    let schema_for_hook = Arc::new(schema.to_string());
    PgPoolOptions::new()
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
        .expect("connect second pool")
}

/// A fresh database records every migration it applied, so a later boot can
/// tell what is already done.
#[tokio::test]
async fn every_migration_is_recorded_in_the_version_table() {
    let Some(fx) = DbFixture::try_new("mig_ver_recorded").await else {
        return;
    };

    // `DbFixture::try_new` already ran the migrations once — the fresh-boot case.
    assert_eq!(
        recorded_migrations(&fx.pool).await,
        migration_files_on_disk(),
        "every migration file must be recorded as applied after a fresh boot"
    );

    fx.cleanup().await;
}

/// The #730 regression: booting again must not re-run migration 021, which
/// drops and re-adds a STORED generated column (full table rewrite + GIN
/// rebuild + one more permanently-consumed `pg_attribute` slot).
#[tokio::test]
async fn second_boot_does_not_drop_and_re_add_the_messages_tsv_column() {
    let Some(fx) = DbFixture::try_new("mig_ver_second_boot").await else {
        return;
    };

    let before = attribute_slots(&fx.pool, fx.schema(), "messages").await;
    run_migrations(&fx.pool)
        .await
        .expect("second boot migrates");
    let after = attribute_slots(&fx.pool, fx.schema(), "messages").await;

    assert_eq!(
        before,
        after,
        "a second boot must apply no migration: `messages` consumed {} extra \
         pg_attribute slot(s), which means 021 dropped and re-added `tsv` again",
        after - before
    );
    assert!(
        has_column(&fx.pool, fx.schema(), "messages", "tsv").await,
        "the FTS column must survive the second boot"
    );

    fx.cleanup().await;
}

/// The transition path: a database migrated before the ledger existed gets
/// the ledger backfilled, keeps its data, and is quiet from then on.
#[tokio::test]
async fn legacy_database_without_a_version_table_is_backfilled() {
    let Some(fx) = DbFixture::try_new("mig_ver_legacy").await else {
        return;
    };

    sqlx::query(
        "INSERT INTO conversations \
            (id, user_id, title, created_at, updated_at, context_summary, compacted_through) \
         VALUES ('c1', 'default', 't', now(), now(), '', 0)",
    )
    .execute(&fx.pool)
    .await
    .expect("seed conversation");
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, user_id, ordinal, role, content) \
         VALUES ('m1', 'c1', 'default', 1, 'user', 'the quick brown fox')",
    )
    .execute(&fx.pool)
    .await
    .expect("seed message");

    simulate_pre_ledger_database(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("transition boot migrates a pre-ledger database");

    assert_eq!(
        recorded_migrations(&fx.pool).await,
        migration_files_on_disk(),
        "the transition boot must record every migration as applied"
    );

    let surviving: i64 = sqlx::query_scalar("SELECT count(*) FROM messages")
        .fetch_one(&fx.pool)
        .await
        .expect("count messages");
    assert_eq!(surviving, 1, "the transition must not lose rows");
    let fts_hits: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM messages WHERE tsv @@ to_tsquery('english', 'fox')",
    )
    .fetch_one(&fx.pool)
    .await
    .expect("run FTS query");
    assert_eq!(fts_hits, 1, "the FTS column must still be populated");

    // And the boot after the transition is a no-op.
    let before = attribute_slots(&fx.pool, fx.schema(), "messages").await;
    run_migrations(&fx.pool)
        .await
        .expect("post-transition boot");
    assert_eq!(
        before,
        attribute_slots(&fx.pool, fx.schema(), "messages").await,
        "once backfilled, later boots must apply nothing"
    );

    fx.cleanup().await;
}

/// A pre-ledger database is not assumed to be at the current head: an install
/// upgrading across releases may never have run the newest migrations, so the
/// transition must apply them rather than mark them applied.
///
/// This is the one guard here that a replay-everything runner also satisfies.
/// It exists to stop the ledger from being backfilled by assumption — marking
/// every migration applied on a database whose last daemon predated some of
/// them silently skips schema an upgrade needs.
#[tokio::test]
async fn legacy_transition_applies_migrations_that_never_ran() {
    let Some(fx) = DbFixture::try_new("mig_ver_legacy_partial").await else {
        return;
    };

    // Roll `knowledge_base` back to its pre-038 shape — the state of a
    // database whose last daemon predated that migration.
    sqlx::query(
        "ALTER TABLE knowledge_base \
             DROP COLUMN IF EXISTS deleted_kind, \
             DROP COLUMN IF EXISTS deleted_reason, \
             DROP COLUMN IF EXISTS superseded_by",
    )
    .execute(&fx.pool)
    .await
    .expect("roll back migration 038");
    simulate_pre_ledger_database(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("transition boot migrates a behind-head database");

    for column in ["deleted_kind", "deleted_reason", "superseded_by"] {
        assert!(
            has_column(&fx.pool, fx.schema(), "knowledge_base", column).await,
            "migration 038 never ran on this database, so the transition must \
             apply it — `knowledge_base.{column}` is missing"
        );
    }

    fx.cleanup().await;
}

/// Empty-input path: the ledger exists but records nothing (an interrupted
/// transition, or a hand-truncated table). Every migration replays, and every
/// migration is idempotent enough to survive it.
#[tokio::test]
async fn empty_version_table_on_a_migrated_database_replays_cleanly() {
    let Some(fx) = DbFixture::try_new("mig_ver_empty").await else {
        return;
    };

    sqlx::query("DELETE FROM schema_migrations")
        .execute(&fx.pool)
        .await
        .expect("empty the ledger");

    run_migrations(&fx.pool)
        .await
        .expect("an empty ledger replays every migration without error");

    assert_eq!(
        recorded_migrations(&fx.pool).await,
        migration_files_on_disk(),
        "the replay must repopulate the ledger"
    );

    fx.cleanup().await;
}

/// Malformed-input path: a ledger row naming a file this build does not carry
/// (a downgrade, or a hand-edited row) is data about someone else's build.
/// It must be ignored, not crash the boot.
#[tokio::test]
async fn unrecognized_recorded_migration_is_ignored() {
    let Some(fx) = DbFixture::try_new("mig_ver_unknown").await else {
        return;
    };

    sqlx::query("INSERT INTO schema_migrations (name) VALUES ($1)")
        .bind("999_from_a_newer_build.sql")
        .execute(&fx.pool)
        .await
        .expect("record an unknown migration");

    run_migrations(&fx.pool)
        .await
        .expect("an unknown ledger row must not fail the boot");

    let still_there: i64 =
        sqlx::query_scalar("SELECT count(*) FROM schema_migrations WHERE name = $1")
            .bind("999_from_a_newer_build.sql")
            .fetch_one(&fx.pool)
            .await
            .expect("count the unknown row");
    assert_eq!(still_there, 1, "an unknown ledger row must be left alone");

    fx.cleanup().await;
}

/// Concurrency path: two daemons booting against one pre-ledger database must
/// serialize on the advisory lock, so the expensive migrations run once
/// between them rather than once each.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_boots_apply_each_migration_once() {
    let Some(fx) = DbFixture::try_new("mig_ver_concurrent").await else {
        return;
    };
    let other = second_pool(fx.schema()).await;

    simulate_pre_ledger_database(&fx.pool).await;
    let before = attribute_slots(&fx.pool, fx.schema(), "messages").await;

    let (first, second) = tokio::join!(run_migrations(&fx.pool), run_migrations(&other));
    first.expect("first concurrent boot migrates");
    second.expect("second concurrent boot migrates");

    let after = attribute_slots(&fx.pool, fx.schema(), "messages").await;
    assert_eq!(
        after - before,
        1,
        "exactly one of the two boots may re-apply 021's drop/re-add of \
         `messages.tsv`; {} slots were consumed, so the boots raced",
        after - before
    );
    assert_eq!(
        recorded_migrations(&fx.pool).await,
        migration_files_on_disk(),
        "both boots must leave a complete ledger"
    );

    other.close().await;
    fx.cleanup().await;
}
