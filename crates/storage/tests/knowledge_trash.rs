//! The knowledge base's soft-delete tombstones are a *trash can* with a
//! lifecycle of its own (issue #657): retained for a configurable window,
//! reaped automatically once past it, and emptiable on demand.
//!
//! Two structural gaps motivate these tests:
//!
//! 1. The TTL reap only ever ran inside the consolidation transaction, so an
//!    instance with dreaming/consolidation disabled accumulated tombstones
//!    forever -- invisible to every read path, but never freed.
//! 2. There was no way to empty the trash on demand, and no way to see how much
//!    was in it (on `adele-prod`, 681 tombstones against 75 live entries).
//!
//! Every reap is per-tenant: one user's sweep must never touch another's rows.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_trash
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_storage::dreaming::{reap_expired_trash, sweep_expired_trash};
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, with_user_id};
use sqlx::PgPool;

const ALICE: &str = "kb-trash-alice";
const BOB: &str = "kb-trash-bob";

/// Boot a fixture in its own schema with migrations applied. `None` when
/// `TEST_DATABASE_URL` is unset, which is how each test pass-skips.
async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("kb657").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

async fn write_entry(store: &PgKnowledgeBaseStore, user: &str, id: &str, content: &str) {
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(id, content, vec!["notes".into()]))
            .await
            .unwrap_or_else(|e| panic!("write {id}: {e}"));
    })
    .await;
}

/// Soft-delete a row the way consolidation does, `age_days` in the past.
async fn soft_delete_aged(pool: &PgPool, id: &str, age_days: i32) {
    let res = sqlx::query(
        "UPDATE knowledge_base \
         SET deleted_at = NOW() - make_interval(days => $2) \
         WHERE id = $1",
    )
    .bind(id)
    .bind(age_days)
    .execute(pool)
    .await
    .expect("soft delete");
    assert_eq!(res.rows_affected(), 1, "soft delete should touch row {id}");
}

/// Rows still physically present, regardless of `deleted_at`.
async fn surviving_ids(pool: &PgPool) -> Vec<String> {
    let rows: Vec<(String,)> = sqlx::query_as("SELECT id FROM knowledge_base ORDER BY id")
        .fetch_all(pool)
        .await
        .expect("probe surviving rows");
    rows.into_iter().map(|(id,)| id).collect()
}

// --- empty trash ------------------------------------------------------------

#[tokio::test]
async fn empty_trash_removes_all_soft_deleted_entries_for_the_user() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "fresh-tombstone", "retired today").await;
    write_entry(&store, ALICE, "old-tombstone", "retired long ago").await;
    soft_delete_aged(&fx.pool, "fresh-tombstone", 0).await;
    soft_delete_aged(&fx.pool, "old-tombstone", 400).await;

    let removed = with_user_id(UserId::new(ALICE), async { store.empty_trash().await })
        .await
        .expect("empty_trash");

    assert_eq!(
        removed, 2,
        "empty trash must reap every soft-deleted entry regardless of age"
    );
    assert!(
        surviving_ids(&fx.pool).await.is_empty(),
        "no tombstone may survive an explicit empty-trash"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn empty_trash_leaves_live_entries_untouched() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "live", "current fact").await;
    write_entry(&store, ALICE, "retired", "superseded fact").await;
    soft_delete_aged(&fx.pool, "retired", 1).await;

    let removed = with_user_id(UserId::new(ALICE), async { store.empty_trash().await })
        .await
        .expect("empty_trash");

    assert_eq!(removed, 1, "only the tombstone is trash");
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["live".to_string()],
        "emptying the trash must never touch a live entry"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn empty_trash_is_scoped_to_the_calling_user() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "alice-retired", "alice trash").await;
    write_entry(&store, BOB, "bob-retired", "bob trash").await;
    soft_delete_aged(&fx.pool, "alice-retired", 1).await;
    soft_delete_aged(&fx.pool, "bob-retired", 1).await;

    let removed = with_user_id(UserId::new(ALICE), async { store.empty_trash().await })
        .await
        .expect("empty_trash");

    assert_eq!(removed, 1, "alice may only reap her own trash");
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["bob-retired".to_string()],
        "one user emptying their trash must not reap another user's tombstones"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn empty_trash_on_an_empty_trash_is_a_no_op() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "live", "current fact").await;

    // Boundary: nothing to reap is a successful zero, not an error.
    let removed = with_user_id(UserId::new(ALICE), async { store.empty_trash().await })
        .await
        .expect("emptying an already-empty trash must succeed");

    assert_eq!(removed, 0, "an empty trash reports zero removals");
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["live".to_string()],
        "a no-op empty-trash must leave the live entry in place"
    );
    fx.cleanup().await;
}

// --- trash count ------------------------------------------------------------

#[tokio::test]
async fn trash_count_reports_only_soft_deleted_entries() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "live-a", "current").await;
    write_entry(&store, ALICE, "live-b", "current").await;
    write_entry(&store, ALICE, "retired-a", "superseded").await;
    write_entry(&store, ALICE, "retired-b", "pruned").await;
    soft_delete_aged(&fx.pool, "retired-a", 0).await;
    soft_delete_aged(&fx.pool, "retired-b", 90).await;

    let count = with_user_id(UserId::new(ALICE), async { store.trash_count().await })
        .await
        .expect("trash_count");

    assert_eq!(
        count, 2,
        "the trash count is the number of soft-deleted entries, live ones excluded"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn trash_count_is_scoped_to_the_calling_user() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "alice-retired", "alice trash").await;
    write_entry(&store, BOB, "bob-retired-1", "bob trash").await;
    write_entry(&store, BOB, "bob-retired-2", "bob trash").await;
    for id in ["alice-retired", "bob-retired-1", "bob-retired-2"] {
        soft_delete_aged(&fx.pool, id, 1).await;
    }

    let count = with_user_id(UserId::new(ALICE), async { store.trash_count().await })
        .await
        .expect("trash_count");

    assert_eq!(
        count, 1,
        "the trash count must not leak another tenant's tombstones"
    );
    fx.cleanup().await;
}

// --- TTL reap ---------------------------------------------------------------

#[tokio::test]
async fn ttl_reap_runs_when_consolidation_is_disabled() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // No consolidation, no LLM, no dreaming: the periodic sweep alone must free
    // expired tombstones, for every user that has any.
    write_entry(&store, ALICE, "alice-expired", "expired").await;
    write_entry(&store, BOB, "bob-expired", "expired").await;
    write_entry(&store, ALICE, "alice-live", "current").await;
    soft_delete_aged(&fx.pool, "alice-expired", 40).await;
    soft_delete_aged(&fx.pool, "bob-expired", 40).await;

    let reaped = sweep_expired_trash(&fx.pool, 30)
        .await
        .expect("sweep_expired_trash");

    assert_eq!(
        reaped, 2,
        "the sweep must reap expired tombstones with consolidation switched off"
    );
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["alice-live".to_string()],
        "only live entries survive the sweep"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn ttl_reap_respects_the_configured_retention() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "ten-days-old", "retired ten days ago").await;
    soft_delete_aged(&fx.pool, "ten-days-old", 10).await;

    // Retention longer than the tombstone's age: it stays.
    let reaped = with_user_id(UserId::new(ALICE), async {
        reap_expired_trash(&fx.pool, 30).await
    })
    .await
    .expect("reap with 30-day retention");
    assert_eq!(reaped, 0, "a tombstone inside the retention window is kept");
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["ten-days-old".to_string()],
        "a 10-day-old tombstone must survive a 30-day retention"
    );

    // Retention shorter than its age: it goes.
    let reaped = with_user_id(UserId::new(ALICE), async {
        reap_expired_trash(&fx.pool, 5).await
    })
    .await
    .expect("reap with 5-day retention");
    assert_eq!(
        reaped, 1,
        "the same tombstone is expired under a 5-day retention"
    );
    assert!(
        surviving_ids(&fx.pool).await.is_empty(),
        "retention is configurable, not pinned to the 30-day default"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn ttl_reap_with_zero_retention_reaps_immediately() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // Boundary: retention 0 means "do not retain" — a tombstone created moments
    // ago is already expired.
    write_entry(&store, ALICE, "just-retired", "retired just now").await;
    write_entry(&store, ALICE, "live", "current").await;
    soft_delete_aged(&fx.pool, "just-retired", 0).await;

    let reaped = with_user_id(UserId::new(ALICE), async {
        reap_expired_trash(&fx.pool, 0).await
    })
    .await
    .expect("reap with zero retention");

    assert_eq!(reaped, 1, "zero retention reaps a fresh tombstone");
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["live".to_string()],
        "zero retention still must not touch live entries"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn ttl_reap_is_scoped_to_the_calling_user() {
    let Some(fx) = fixture().await else {
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, ALICE, "alice-expired", "expired").await;
    write_entry(&store, BOB, "bob-expired", "expired").await;
    soft_delete_aged(&fx.pool, "alice-expired", 40).await;
    soft_delete_aged(&fx.pool, "bob-expired", 40).await;

    let reaped = with_user_id(UserId::new(ALICE), async {
        reap_expired_trash(&fx.pool, 30).await
    })
    .await
    .expect("reap");

    assert_eq!(reaped, 1, "a per-user reap only sees its own partition");
    assert_eq!(
        surviving_ids(&fx.pool).await,
        vec!["bob-expired".to_string()],
        "one user's TTL reap must never free another user's tombstones"
    );
    fx.cleanup().await;
}
