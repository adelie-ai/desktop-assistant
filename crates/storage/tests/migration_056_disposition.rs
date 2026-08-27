//! Migration 056 widens `knowledge_base.deleted_kind` /
//! `knowledge_base.deleted_reason` into `disposition` / `disposition_reason`,
//! decoupled from `deleted_at` (#893).
//!
//! Two things need proving, and neither is provable by reading the SQL:
//!
//! * The backfill maps old data correctly. A merge tombstone's `deleted_kind
//!   = 'merge'` must become `disposition = 'superseded'`; a prune
//!   tombstone's `'prune'` must become `'trivial'` with its stated reason
//!   intact; a live row with no kind at all must become `'active'`. The
//!   suite rolls a freshly migrated database back to the pre-056 shape,
//!   seeds rows in that shape, replays only migration 056, and reads the
//!   result.
//! * The new CHECK constraints actually refuse what they claim to. A
//!   constraint nothing exercises is a constraint nobody has proven works.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test migration_056_disposition
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_storage::run_migrations;
use sqlx::PgPool;

/// Roll a freshly migrated (through migration 056) `knowledge_base` back to
/// the shape migration 056 expects to find: `deleted_kind` / `deleted_reason`,
/// nullable, no CHECK on the new vocabulary. Then drop migration 056's ledger
/// row, so the next [`run_migrations`] call replays only that one migration.
///
/// This is the same technique `migration_versioning.rs`'s legacy-transition
/// tests use, scoped to a single migration instead of the whole ledger, so
/// the backfill this migration performs can be exercised on data shaped the
/// way a real pre-056 database holds it.
async fn rollback_to_pre_056(pool: &PgPool) {
    sqlx::query(
        "ALTER TABLE knowledge_base \
             DROP CONSTRAINT IF EXISTS knowledge_base_disposition_chk, \
             DROP CONSTRAINT IF EXISTS knowledge_base_superseded_by_chk",
    )
    .execute(pool)
    .await
    .expect("drop migration 056's constraints");

    sqlx::query("ALTER TABLE knowledge_base RENAME COLUMN disposition TO deleted_kind")
        .execute(pool)
        .await
        .expect("rename disposition back to deleted_kind");
    sqlx::query("ALTER TABLE knowledge_base RENAME COLUMN disposition_reason TO deleted_reason")
        .execute(pool)
        .await
        .expect("rename disposition_reason back to deleted_reason");

    sqlx::query(
        "ALTER TABLE knowledge_base \
             ALTER COLUMN deleted_kind DROP NOT NULL, \
             ALTER COLUMN deleted_kind DROP DEFAULT",
    )
    .execute(pool)
    .await
    .expect("relax deleted_kind back to its pre-056 shape");

    sqlx::query("DELETE FROM schema_migrations WHERE name = '056_kb_disposition.sql'")
        .execute(pool)
        .await
        .expect("unrecord migration 056 so it replays");
}

/// A pre-056-shaped row's tombstone provenance: the old `deleted_kind` /
/// `deleted_reason` / `superseded_by` triple, plus how long ago it was
/// deleted. The default (every field `None`) describes a live row.
#[derive(Default)]
struct Tombstone<'a> {
    days_ago: Option<i32>,
    kind: Option<&'a str>,
    reason: Option<&'a str>,
    superseded_by: Option<&'a str>,
}

async fn insert_row(
    pool: &PgPool,
    id: &str,
    user_id: &str,
    content: &str,
    tombstone: Tombstone<'_>,
) {
    sqlx::query(
        "INSERT INTO knowledge_base \
             (id, user_id, content, deleted_at, deleted_kind, deleted_reason, superseded_by) \
         VALUES ($1, $2, $3, \
                 CASE WHEN $4::int IS NULL THEN NULL ELSE NOW() - make_interval(days => $4) END, \
                 $5, $6, $7)",
    )
    .bind(id)
    .bind(user_id)
    .bind(content)
    .bind(tombstone.days_ago)
    .bind(tombstone.kind)
    .bind(tombstone.reason)
    .bind(tombstone.superseded_by)
    .execute(pool)
    .await
    .expect("seed a pre-056-shaped knowledge_base row");
}

async fn disposition(pool: &PgPool, id: &str) -> String {
    sqlx::query_scalar("SELECT disposition FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read disposition")
}

async fn disposition_reason(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT disposition_reason FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read disposition_reason")
}

async fn superseded_by(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT superseded_by FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read superseded_by")
}

#[tokio::test]
async fn migration_056_maps_merge_tombstones_to_superseded() {
    let Some(fx) = support::DbFixture::try_new("mig056_merge").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    rollback_to_pre_056(&fx.pool).await;
    insert_row(
        &fx.pool,
        "kb-merge-member",
        "u1",
        "a near-duplicate fact",
        Tombstone {
            days_ago: Some(1),
            kind: Some("merge"),
            superseded_by: Some("kb-canonical"),
            ..Default::default()
        },
    )
    .await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 056 replays");

    assert_eq!(
        disposition(&fx.pool, "kb-merge-member").await,
        "superseded",
        "a merge tombstone's deleted_kind maps to the superseded disposition"
    );
    assert_eq!(
        superseded_by(&fx.pool, "kb-merge-member").await.as_deref(),
        Some("kb-canonical"),
        "the merge's successor link is carried forward unchanged"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn migration_056_maps_prune_tombstones_to_trivial_and_keeps_the_reason() {
    let Some(fx) = support::DbFixture::try_new("mig056_prune").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    rollback_to_pre_056(&fx.pool).await;
    insert_row(
        &fx.pool,
        "kb-pruned",
        "u1",
        "circumstantial detail",
        Tombstone {
            days_ago: Some(1),
            kind: Some("prune"),
            reason: Some("mattered only in the moment"),
            ..Default::default()
        },
    )
    .await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 056 replays");

    assert_eq!(
        disposition(&fx.pool, "kb-pruned").await,
        "trivial",
        "a prune tombstone's deleted_kind maps to the trivial disposition"
    );
    assert_eq!(
        disposition_reason(&fx.pool, "kb-pruned").await.as_deref(),
        Some("mattered only in the moment"),
        "the model's stated reason survives the rename unchanged"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn migration_056_backfills_live_rows_to_active() {
    let Some(fx) = support::DbFixture::try_new("mig056_live").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    rollback_to_pre_056(&fx.pool).await;
    insert_row(
        &fx.pool,
        "kb-live",
        "u1",
        "an ordinary fact",
        Tombstone::default(),
    )
    .await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 056 replays");

    assert_eq!(
        disposition(&fx.pool, "kb-live").await,
        "active",
        "a live row with no recorded kind defaults to active"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn disposition_check_rejects_an_unknown_value() {
    let Some(fx) = support::DbFixture::try_new("mig056_bad_value").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    let err = sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition) \
         VALUES ('kb-bad', 'u1', 'x', 'archived')",
    )
    .execute(&fx.pool)
    .await
    .expect_err("a spelling the CHECK constraint does not list must be refused");
    assert!(
        err.to_string().contains("knowledge_base_disposition_chk"),
        "the refusal should name the constraint that fired: {err}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn superseded_without_a_successor_is_rejected_by_the_schema() {
    let Some(fx) = support::DbFixture::try_new("mig056_no_successor").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    let err = sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-orphan', 'u1', 'x', 'superseded', NULL)",
    )
    .execute(&fx.pool)
    .await
    .expect_err("a superseded row naming no successor must be refused");
    assert!(
        err.to_string().contains("knowledge_base_superseded_by_chk"),
        "the refusal should name the constraint that fired: {err}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn redundant_without_a_successor_is_rejected_by_the_schema() {
    // The same constraint covers both dispositions that resolve through a
    // link; this is the sibling case to the test above, not a repeat of it.
    let Some(fx) = support::DbFixture::try_new("mig056_redundant_no_successor").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    let err = sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-orphan', 'u1', 'x', 'redundant', NULL)",
    )
    .execute(&fx.pool)
    .await
    .expect_err("a redundant row naming no successor must be refused");
    assert!(
        err.to_string().contains("knowledge_base_superseded_by_chk"),
        "the refusal should name the constraint that fired: {err}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn an_active_row_may_not_name_a_successor() {
    // The constraint is an equivalence, not a one-way implication: a
    // successor link on a row that is not superseded or redundant is just as
    // much a lie as a superseded row with no link.
    let Some(fx) = support::DbFixture::try_new("mig056_active_with_successor").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    let err = sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-a', 'u1', 'x', 'active', 'kb-b')",
    )
    .execute(&fx.pool)
    .await
    .expect_err("an active row naming a successor must be refused");
    assert!(
        err.to_string().contains("knowledge_base_superseded_by_chk"),
        "the refusal should name the constraint that fired: {err}"
    );

    fx.cleanup().await;
}
