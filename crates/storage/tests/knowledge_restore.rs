//! Restore and trash search (#710).
//!
//! Before this suite, nothing in the daemon, a client, or a tool could ever
//! clear `deleted_at`: a soft-deleted row was recoverable in principle, by
//! hand, with raw SQL against Postgres, and unrecoverable in practice once
//! the retention reap freed it. This suite pins the pair that closes the
//! gap: [`restore_entry`] brings a tombstone back, and [`search_trash`] finds
//! one by full text without knowing its id first.
//!
//! Each test is named for the acceptance criterion it holds, so a failing run
//! names the unmet requirement.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_restore
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::{KnowledgeBaseStore, RestoreOutcome};
use desktop_assistant_storage::dreaming::{restore_entry, search_trash};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, with_user_id};
use sqlx::PgPool;

const ALICE: &str = "kb710-alice";
const BOB: &str = "kb710-bob";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("kb710").await;
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

/// Soft-delete a row the way consolidation or a non-person delete does,
/// stamping whatever disposition columns the caller wants.
async fn tombstone(
    pool: &PgPool,
    id: &str,
    disposition: &str,
    disposition_reason: Option<&str>,
    superseded_by: Option<&str>,
) {
    let res = sqlx::query(
        "UPDATE knowledge_base \
         SET deleted_at = NOW(), \
             disposition = $2, \
             disposition_reason = $3, \
             superseded_by = $4 \
         WHERE id = $1",
    )
    .bind(id)
    .bind(disposition)
    .bind(disposition_reason)
    .bind(superseded_by)
    .execute(pool)
    .await
    .expect("tombstone row");
    assert_eq!(res.rows_affected(), 1, "tombstone should touch row {id}");
}

/// Clear a row's embedding, the way #683's sweep does when the embedding
/// model changes underneath a stale vector.
async fn clear_embedding(pool: &PgPool, id: &str) {
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = NULL, embedding_model = NULL, embeddings_updated_at = NOW() \
         WHERE id = $1",
    )
    .bind(id)
    .execute(pool)
    .await
    .expect("clear embedding");
}

/// Whether `id` currently matches the embedding backfill's own row-selection
/// predicate (`crate::embedding_backfill::backfill_knowledge_embeddings`),
/// for the model named `model`. Repeats that predicate rather than importing
/// it, so a change to one is caught by the other drifting.
async fn needs_backfill(pool: &PgPool, id: &str, model: &str) -> bool {
    sqlx::query_scalar(
        "SELECT deleted_at IS NULL \
                AND (embedding_model IS NULL \
                  OR embedding_model != $2 \
                  OR embeddings_updated_at IS NULL \
                  OR embeddings_updated_at < updated_at) \
         FROM knowledge_base WHERE id = $1",
    )
    .bind(id)
    .bind(model)
    .fetch_one(pool)
    .await
    .expect("read backfill eligibility")
}

async fn row_exists(pool: &PgPool, id: &str) -> bool {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("count kb row");
    count > 0
}

async fn is_deleted(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT deleted_at IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read deleted_at")
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_soft_deleted_entry_can_be_restored_and_is_searchable_again() {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());

    write_entry(
        &store,
        ALICE,
        "restore-me",
        "the office wifi password rotates monthly",
    )
    .await;
    tombstone(pool, "restore-me", "trivial", Some("looked stale"), None).await;

    let outcome = with_user_id(UserId::new(ALICE), async {
        restore_entry(pool, "restore-me").await
    })
    .await
    .expect("restore succeeds");
    assert_eq!(outcome, RestoreOutcome::Restored);

    // Searchable again through the ordinary text-search path, exactly as an
    // entry that was never touched would be.
    let found = with_user_id(UserId::new(ALICE), async {
        store.search_text("wifi password", None, 10).await
    })
    .await
    .expect("search");
    assert!(
        found.iter().any(|e| e.id == "restore-me"),
        "a restored entry must be findable by ordinary search again"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn restore_is_user_scoped_a_second_tenant_cannot_restore_anothers_tombstone() {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());

    write_entry(&store, ALICE, "alices-secret", "alice's own retired fact").await;
    tombstone(pool, "alices-secret", "trivial", None, None).await;

    // Bob asks to restore an id he has never seen.
    let outcome = with_user_id(UserId::new(BOB), async {
        restore_entry(pool, "alices-secret").await
    })
    .await
    .expect("restore call succeeds even when it finds nothing to restore");
    assert_eq!(
        outcome,
        RestoreOutcome::NoLongerExists,
        "another tenant's id must read as though it names nothing at all"
    );

    assert!(
        is_deleted(pool, "alices-secret").await,
        "bob's attempt must not have touched alice's tombstone"
    );

    // Alice can still restore her own.
    let outcome = with_user_id(UserId::new(ALICE), async {
        restore_entry(pool, "alices-secret").await
    })
    .await
    .expect("restore succeeds");
    assert_eq!(outcome, RestoreOutcome::Restored);

    fx.cleanup().await;
}

#[tokio::test]
async fn restoring_an_entry_with_a_cleared_vector_reembeds_rather_than_resurrecting_a_wrong_dimension_vector()
 {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());

    write_entry(
        &store,
        ALICE,
        "stale-vector",
        "a fact embedded by a retired model",
    )
    .await;
    tombstone(pool, "stale-vector", "trivial", None, None).await;
    // Simulate #683's sweep clearing the vector on a tombstone whose model
    // was superseded while it sat in the trash.
    clear_embedding(pool, "stale-vector").await;

    let outcome = with_user_id(UserId::new(ALICE), async {
        restore_entry(pool, "stale-vector").await
    })
    .await
    .expect("restore succeeds");
    assert_eq!(outcome, RestoreOutcome::Restored);

    // Restore must not write a vector back in; it only clears the tombstone
    // columns. The row must therefore still read as needing the backfill,
    // for the model the daemon currently embeds with - never resurrecting
    // whatever (possibly wrong-dimension) vector it held before.
    assert!(
        needs_backfill(pool, "stale-vector", "current-model").await,
        "a restored entry with a cleared vector must stay eligible for the embedding backfill"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn restore_clears_the_delete_provenance_columns() {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());

    write_entry(
        &store,
        ALICE,
        "merged-away",
        "a fact absorbed into a merge synthesis",
    )
    .await;
    // A merge tombstone under migration 056's mapping: disposition
    // 'redundant' (or 'superseded'), a stated reason, and a successor id -
    // the schema's own `knowledge_base_superseded_by_chk` requires exactly
    // this pairing, so this is the shape a real merge tombstone has. It
    // covers the case with the most to lose: restore must clear the
    // successor link along with the disposition and the reason, not only the
    // simpler prune shape (disposition alone, no successor).
    tombstone(
        pool,
        "merged-away",
        "redundant",
        Some("absorbed into a merge synthesis"),
        Some("some-other-row-id"),
    )
    .await;

    let outcome = with_user_id(UserId::new(ALICE), async {
        restore_entry(pool, "merged-away").await
    })
    .await
    .expect("restore succeeds");
    assert_eq!(outcome, RestoreOutcome::Restored);

    let (disposition, reason, successor): (String, Option<String>, Option<String>) =
        sqlx::query_as(
            "SELECT disposition, disposition_reason, superseded_by \
             FROM knowledge_base WHERE id = $1",
        )
        .bind("merged-away")
        .fetch_one(pool)
        .await
        .expect("read restored row");

    // The disposition decision: restore always resets to `active`, whatever
    // verdict the tombstone carried. Leaving a restored row demoted (still
    // 'trivial', still pointing at a successor) would mean a person who went
    // to the trouble of finding and restoring it got back an entry flagged
    // "not worth keeping" the moment it reappeared - the opposite of what
    // restoring it means.
    assert_eq!(
        disposition, "active",
        "restore must clear the disposition, not just deleted_at"
    );
    assert_eq!(
        reason, None,
        "the stated reason for retiring the row must not survive a restore"
    );
    assert_eq!(
        successor, None,
        "a restored row must not still point at a successor"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_reaped_entry_reports_absence_clearly() {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;

    // No row was ever written at this id: the same shape a hard-reaped
    // tombstone leaves behind.
    let outcome = with_user_id(UserId::new(ALICE), async {
        restore_entry(pool, "never-existed").await
    })
    .await
    .expect("restore call succeeds even when it finds nothing to restore");
    assert_eq!(outcome, RestoreOutcome::NoLongerExists);

    fx.cleanup().await;
}

#[tokio::test]
async fn restoring_a_still_live_entry_reports_it_was_never_in_the_trash() {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());

    write_entry(&store, ALICE, "still-live", "never retired").await;

    let outcome = with_user_id(UserId::new(ALICE), async {
        restore_entry(pool, "still-live").await
    })
    .await
    .expect("restore call succeeds even when there is nothing to restore");
    // Distinct from `NoLongerExists`: the id names a real row this user owns,
    // it is simply not in the trash - a different mistake than asking for
    // something that is truly gone.
    assert_eq!(outcome, RestoreOutcome::NotInTrash);
    assert!(row_exists(pool, "still-live").await);
    assert!(!is_deleted(pool, "still-live").await);

    fx.cleanup().await;
}

#[tokio::test]
async fn tombstones_are_searchable_by_full_text_without_knowing_the_id() {
    let Some(fx) = fixture().await else {
        return;
    };
    let pool = &fx.pool;
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());

    write_entry(
        &store,
        ALICE,
        "trash-hit",
        "the printer driver needs a manual restart",
    )
    .await;
    tombstone(
        pool,
        "trash-hit",
        "trivial",
        Some("looked like noise"),
        None,
    )
    .await;
    write_entry(
        &store,
        ALICE,
        "trash-miss",
        "an unrelated retired fact about coffee",
    )
    .await;
    tombstone(pool, "trash-miss", "trivial", None, None).await;
    // A live entry sharing the same words must not appear in a trash search.
    write_entry(
        &store,
        ALICE,
        "still-live-printer",
        "the printer driver is up to date",
    )
    .await;

    let hits = with_user_id(UserId::new(ALICE), async {
        search_trash(pool, "printer driver", 10).await
    })
    .await
    .expect("search_trash succeeds");

    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    assert!(
        ids.contains(&"trash-hit"),
        "the matching tombstone must be found: {ids:?}"
    );
    assert!(
        !ids.contains(&"trash-miss"),
        "a tombstone that does not match the query must not appear: {ids:?}"
    );
    assert!(
        !ids.contains(&"still-live-printer"),
        "a live row must never appear in a trash search: {ids:?}"
    );

    let hit = hits.iter().find(|h| h.id == "trash-hit").expect("found");
    assert_eq!(hit.disposition.as_str(), "trivial");
    assert_eq!(hit.disposition_reason.as_deref(), Some("looked like noise"));
    assert!(
        !hit.deleted_at.is_empty(),
        "a trash result must carry when it was retired"
    );

    fx.cleanup().await;
}
