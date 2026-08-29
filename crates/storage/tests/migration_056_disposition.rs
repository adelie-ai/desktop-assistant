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

async fn is_tombstoned(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar::<_, bool>("SELECT deleted_at IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read deleted_at")
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
async fn migration_056_backfills_a_merge_tombstone_with_no_successor_to_active() {
    // A merge tombstone whose successor id was never recorded cannot become
    // 'superseded': that disposition asserts a link the row does not have,
    // and the constraint the migration adds refuses it. This exercises the
    // backfill's second merge arm, which nothing else in this suite seeds -
    // every other merge fixture carries a successor id and only ever
    // reaches the first arm.
    let Some(fx) = support::DbFixture::try_new("mig056_merge_no_successor").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    rollback_to_pre_056(&fx.pool).await;
    insert_row(
        &fx.pool,
        "kb-merge-orphan",
        "u1",
        "a near-duplicate fact whose successor was never recorded",
        Tombstone {
            days_ago: Some(1),
            kind: Some("merge"),
            ..Default::default()
        },
    )
    .await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 056 replays");

    assert_eq!(
        disposition(&fx.pool, "kb-merge-orphan").await,
        "active",
        "a merge tombstone with no successor id must not become superseded"
    );
    assert!(
        is_tombstoned(&fx.pool, "kb-merge-orphan").await,
        "the backfill must not touch deleted_at - the row stays in the trash"
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
async fn the_constraint_still_refuses_superseded_without_a_successor() {
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
async fn the_constraint_still_refuses_redundant_without_a_successor() {
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
async fn the_constraint_permits_a_refuted_entry_naming_its_successor() {
    // The constraint is a one-way implication, not an equivalence: a
    // disposition other than superseded/redundant is free to name a
    // successor too. 'refuted' is exactly the case the consolidation
    // contract asks the model to produce (see the "Rules the store
    // enforces" section of the dreaming prompt).
    let Some(fx) = support::DbFixture::try_new("mig056_refuted_with_successor").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition) \
         VALUES ('kb-successor', 'u1', 'the corrected fact', 'active')",
    )
    .execute(&fx.pool)
    .await
    .expect("seed the successor row");

    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-refuted', 'u1', 'the wrong fact', 'refuted', 'kb-successor')",
    )
    .execute(&fx.pool)
    .await
    .expect("a refuted row naming its successor must be permitted");

    fx.cleanup().await;
}

#[tokio::test]
async fn the_backfill_preserves_a_successor_link_whatever_the_tombstone_kind() {
    // A prune tombstone (kind 'prune') is not one of the two kinds the old
    // backfill ever mapped to 'superseded', but a superseded_by id on such a
    // row is still real data, not noise: it must survive the backfill
    // untouched, whatever disposition the kind maps to. The same holds for a
    // tombstone with no recorded kind at all.
    let Some(fx) = support::DbFixture::try_new("mig056_link_survives_kind").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    rollback_to_pre_056(&fx.pool).await;

    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, deleted_at) \
         VALUES ('kb-successor', 'u1', 'the row that absorbed it', NOW())",
    )
    .execute(&fx.pool)
    .await
    .expect("seed a successor row");
    insert_row(
        &fx.pool,
        "kb-pruned-linked",
        "u1",
        "a fact judged not worth keeping, but linked anyway",
        Tombstone {
            days_ago: Some(1),
            kind: Some("prune"),
            reason: Some("mattered only in the moment"),
            superseded_by: Some("kb-successor"),
        },
    )
    .await;
    insert_row(
        &fx.pool,
        "kb-unkinded-linked",
        "u1",
        "a tombstone from before deleted_kind ever recorded anything",
        Tombstone {
            days_ago: Some(1),
            superseded_by: Some("kb-successor"),
            ..Default::default()
        },
    )
    .await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 056 replays");

    assert_eq!(
        disposition(&fx.pool, "kb-pruned-linked").await,
        "trivial",
        "the prune kind still maps to trivial"
    );
    assert_eq!(
        superseded_by(&fx.pool, "kb-pruned-linked").await.as_deref(),
        Some("kb-successor"),
        "a prune tombstone's successor link must survive the backfill"
    );
    assert_eq!(
        disposition(&fx.pool, "kb-unkinded-linked").await,
        "active",
        "a kindless tombstone still maps to active"
    );
    assert_eq!(
        superseded_by(&fx.pool, "kb-unkinded-linked")
            .await
            .as_deref(),
        Some("kb-successor"),
        "a kindless tombstone's successor link must survive the backfill too"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_row_that_would_violate_the_constraint_fails_the_migration_by_name() {
    // A merge tombstone with no successor id backfills to 'active', which
    // this migration's own logic guarantees never violates the constraint
    // it adds. To exercise the pre-flight diagnostic itself, this test
    // instead puts a row directly into the one shape the diagnostic exists
    // to catch -- disposition already 'superseded' with no superseded_by --
    // by writing it while the guarding constraints are absent, standing in
    // for however such a row might reach this shape (hand-edited data, or a
    // future change to the backfill that stops preserving the invariant).
    let Some(fx) = support::DbFixture::try_new("mig056_preflight_fires").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    sqlx::query(
        "ALTER TABLE knowledge_base \
             DROP CONSTRAINT IF EXISTS knowledge_base_disposition_chk, \
             DROP CONSTRAINT IF EXISTS knowledge_base_superseded_by_chk",
    )
    .execute(&fx.pool)
    .await
    .expect("drop migration 056's constraints so a violating row can be written");

    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, disposition, superseded_by) \
         VALUES ('kb-violator', 'u1', 'x', 'superseded', NULL)",
    )
    .execute(&fx.pool)
    .await
    .expect("seed a row shaped to violate the constraint once it is re-added");

    sqlx::query("DELETE FROM schema_migrations WHERE name = '056_kb_disposition.sql'")
        .execute(&fx.pool)
        .await
        .expect("unrecord migration 056 so it replays");

    let err = run_migrations(&fx.pool)
        .await
        .expect_err("a violating row must fail the migration rather than silently pass");
    let message = err.to_string();
    assert!(
        message.contains("knowledge_base_superseded_by_chk"),
        "the failure should name the constraint that would be violated: {message}"
    );
    assert!(
        message.contains("1 row(s) would violate knowledge_base_superseded_by_chk"),
        "the failure should name the offending row count in the diagnostic's own words: \
         {message}"
    );

    fx.cleanup().await;
}
