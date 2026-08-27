//! Migration 058 revives merge-member tombstones -- the recoverable half of
//! the migration question (#694).
//!
//! Consolidation's old destructive verb soft-deleted a merge's losing
//! entries and recorded each one's successor in `superseded_by`. Migration
//! 058 brings those rows back where the disk still holds enough to tell what
//! happened: a member whose successor still exists comes back `superseded`
//! (search resolves through the link); a member whose successor was
//! hard-reaped comes back `active`, because it is the only surviving copy of
//! its content. A prune tombstone and a pre-038 tombstone with no recorded
//! kind are outside the migration's filter and must be left exactly as they
//! are -- reviving them would either overturn a real judgement (prune) or
//! guess at linkage that was never recorded (pre-038).
//!
//! Every case here seeds a fixture shaped the way the real column already
//! looks after migration 056 (there is no schema change to roll back), then
//! rewinds only migration 058's ledger row so `run_migrations` replays it in
//! isolation.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test migration_058_revive_tombstones
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use chrono::{DateTime, Utc};
use desktop_assistant_storage::run_migrations;
use sqlx::PgPool;

const MIGRATION_058: &str = "058_revive_merge_tombstones.sql";

/// Delete migration 058's ledger row so the next [`run_migrations`] call
/// replays only that migration against the fixture's already-current schema.
/// Mirrors `migration_056_disposition.rs`'s `rollback_to_pre_056`, scoped to
/// data rather than schema: 058 changes no column, so there is nothing to
/// roll back beyond un-recording it.
async fn rewind_058(pool: &PgPool) {
    sqlx::query("DELETE FROM schema_migrations WHERE name = $1")
        .bind(MIGRATION_058)
        .execute(pool)
        .await
        .expect("unrecord migration 058 so it replays");
}

/// Seed one `knowledge_base` row in the shape migration 058 reads: an id, a
/// disposition, an optional tombstone age, and an optional successor link.
/// `days_ago = None` means the row is live (`deleted_at IS NULL`).
async fn insert_row(
    pool: &PgPool,
    id: &str,
    disposition: &str,
    days_ago: Option<i32>,
    superseded_by: Option<&str>,
) {
    sqlx::query(
        "INSERT INTO knowledge_base \
             (id, user_id, content, disposition, superseded_by, deleted_at) \
         VALUES ($1, 'u1', 'seed content', $2, $3, \
                 CASE WHEN $4::int IS NULL THEN NULL ELSE NOW() - make_interval(days => $4) END)",
    )
    .bind(id)
    .bind(disposition)
    .bind(superseded_by)
    .bind(days_ago)
    .execute(pool)
    .await
    .expect("seed a knowledge_base row");
}

async fn deleted_at(pool: &PgPool, id: &str) -> Option<DateTime<Utc>> {
    sqlx::query_scalar("SELECT deleted_at FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read deleted_at")
}

async fn disposition(pool: &PgPool, id: &str) -> String {
    sqlx::query_scalar("SELECT disposition FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read disposition")
}

async fn superseded_by(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT superseded_by FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read superseded_by")
}

#[tokio::test]
async fn a_merge_tombstone_with_a_living_successor_revives_as_superseded() {
    let Some(fx) = support::DbFixture::try_new("mig058_living").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    insert_row(&fx.pool, "kb-canonical", "active", None, None).await;
    insert_row(
        &fx.pool,
        "kb-member",
        "superseded",
        Some(1),
        Some("kb-canonical"),
    )
    .await;
    rewind_058(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 058 replays");

    assert!(
        deleted_at(&fx.pool, "kb-member").await.is_none(),
        "the member must come back live once its successor still exists"
    );
    assert_eq!(
        disposition(&fx.pool, "kb-member").await,
        "superseded",
        "a revived member with a living successor stays superseded, so search \
         still resolves through the link"
    );
    assert_eq!(
        superseded_by(&fx.pool, "kb-member").await.as_deref(),
        Some("kb-canonical"),
        "the successor link must survive the revival unchanged"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_merge_tombstone_with_a_reaped_successor_revives_as_active() {
    let Some(fx) = support::DbFixture::try_new("mig058_reaped").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    // No row named "kb-hard-reaped" exists at all: the successor was
    // permanently removed, which is exactly what a dangling `superseded_by`
    // records.
    insert_row(
        &fx.pool,
        "kb-member",
        "superseded",
        Some(1),
        Some("kb-hard-reaped"),
    )
    .await;
    rewind_058(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 058 replays");

    assert!(
        deleted_at(&fx.pool, "kb-member").await.is_none(),
        "the member must come back live even though its successor is gone"
    );
    assert_eq!(
        disposition(&fx.pool, "kb-member").await,
        "active",
        "with no living successor the member is the only surviving copy, so \
         it stands on its own as active"
    );
    assert_eq!(
        superseded_by(&fx.pool, "kb-member").await,
        None,
        "a dangling successor id must be cleared, not asserted as a link \
         this database can still follow"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_prune_tombstone_stays_in_the_trash() {
    let Some(fx) = support::DbFixture::try_new("mig058_prune").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    insert_row(&fx.pool, "kb-pruned", "trivial", Some(1), None).await;
    rewind_058(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 058 replays");

    assert!(
        deleted_at(&fx.pool, "kb-pruned").await.is_some(),
        "a prune tombstone was a judgement of worthlessness, not a \
         relocation -- it must stay deleted"
    );
    assert_eq!(
        disposition(&fx.pool, "kb-pruned").await,
        "trivial",
        "a prune tombstone's disposition is untouched by this migration"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_pre_038_tombstone_with_null_disposition_is_left_untouched() {
    let Some(fx) = support::DbFixture::try_new("mig058_pre038").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    // Migration 056 backfills a pre-038 tombstone's NULL `deleted_kind` to
    // `disposition = 'active'`, because no judgement was ever captured for
    // it -- `deleted_at` alone still says the row is in the trash. This
    // fixture recreates exactly that shape.
    insert_row(&fx.pool, "kb-legacy", "active", Some(1), None).await;
    rewind_058(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 058 replays");

    assert!(
        deleted_at(&fx.pool, "kb-legacy").await.is_some(),
        "a pre-038 tombstone carries no recorded merge linkage, so this \
         migration must not guess at it -- the row stays deleted"
    );
    assert_eq!(
        disposition(&fx.pool, "kb-legacy").await,
        "active",
        "the backfilled disposition is untouched"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_live_row_is_left_untouched() {
    let Some(fx) = support::DbFixture::try_new("mig058_live").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    insert_row(&fx.pool, "kb-live", "active", None, None).await;
    rewind_058(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("migration 058 replays");

    assert!(
        deleted_at(&fx.pool, "kb-live").await.is_none(),
        "a row that was never deleted must not be touched by a revival \
         migration"
    );
    assert_eq!(disposition(&fx.pool, "kb-live").await, "active");

    fx.cleanup().await;
}

#[tokio::test]
async fn migration_058_is_idempotent() {
    let Some(fx) = support::DbFixture::try_new("mig058_idempotent").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    insert_row(&fx.pool, "kb-canonical", "active", None, None).await;
    insert_row(
        &fx.pool,
        "kb-living-member",
        "superseded",
        Some(1),
        Some("kb-canonical"),
    )
    .await;
    insert_row(
        &fx.pool,
        "kb-reaped-member",
        "superseded",
        Some(1),
        Some("kb-hard-reaped"),
    )
    .await;
    insert_row(&fx.pool, "kb-pruned", "trivial", Some(1), None).await;
    rewind_058(&fx.pool).await;

    run_migrations(&fx.pool)
        .await
        .expect("first replay of migration 058");
    let after_first = (
        deleted_at(&fx.pool, "kb-living-member").await,
        disposition(&fx.pool, "kb-living-member").await,
        deleted_at(&fx.pool, "kb-reaped-member").await,
        disposition(&fx.pool, "kb-reaped-member").await,
        superseded_by(&fx.pool, "kb-reaped-member").await,
        deleted_at(&fx.pool, "kb-pruned").await,
        disposition(&fx.pool, "kb-pruned").await,
    );

    // A second call is an ordinary boot with an unmodified ledger: 058 is
    // already recorded, so `run_migrations` will not even attempt it again.
    // Rewinding a second time and replaying proves the SQL itself is a
    // no-op on rows it already touched, which is the actual property this
    // test names -- not just that the runner skips a recorded migration.
    rewind_058(&fx.pool).await;
    run_migrations(&fx.pool)
        .await
        .expect("second replay of migration 058");
    let after_second = (
        deleted_at(&fx.pool, "kb-living-member").await,
        disposition(&fx.pool, "kb-living-member").await,
        deleted_at(&fx.pool, "kb-reaped-member").await,
        disposition(&fx.pool, "kb-reaped-member").await,
        superseded_by(&fx.pool, "kb-reaped-member").await,
        deleted_at(&fx.pool, "kb-pruned").await,
        disposition(&fx.pool, "kb-pruned").await,
    );

    assert_eq!(
        after_first, after_second,
        "replaying migration 058 against rows it already revived must \
         change nothing"
    );

    fx.cleanup().await;
}
