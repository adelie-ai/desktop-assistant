//! Soft-deleted knowledge-base entries must be invisible to every read path
//! and skipped by the embedding pipeline (issue #656).
//!
//! Consolidation soft-deletes an entry when it is superseded or pruned, but no
//! read path filtered `deleted_at`, so retired facts stayed fully searchable
//! and were handed back to the assistant as current. On `adele-prod` that was
//! 681 tombstones against 75 live entries -- roughly 90% of everything the
//! knowledge base could return had already been retired.
//!
//! The embedding pipeline had the same split: `invalidate_all_knowledge_
//! embeddings` (the "Recalculate Embeddings" button) skips soft-deleted rows
//! and says so, while `invalidate_stale_embeddings` and the backfill
//! row-selection query did not -- so every model change re-embedded the
//! tombstones too.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db -- --test knowledge_soft_delete
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::{
    KnowledgeBaseStore, KnowledgeListQuery, ListOrder, ListOrderOpt,
};
use desktop_assistant_storage::embedding_backfill::{
    BackfillEmbedFn, backfill_knowledge_embeddings, invalidate_stale_embeddings,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, run_migrations, with_user_id};
use pgvector::Vector;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

const USER: &str = "kb-owner";

/// Stamp a `vector[]` embedding onto a row. Writes never embed inline (the
/// background backfill does), so tests populate the column directly to make
/// the search vector branch actually run.
async fn set_embedding(pool: &PgPool, id: &str, chunks: Vec<Vec<f32>>) {
    let vecs: Vec<Vector> = chunks.into_iter().map(Vector::from).collect();
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = 'model-A', \
             embeddings_updated_at = NOW() \
         WHERE id = $2",
    )
    .bind(&vecs)
    .bind(id)
    .execute(pool)
    .await
    .expect("stamp embedding");
}

/// Soft-delete a row the way consolidation does.
async fn soft_delete(pool: &PgPool, id: &str) {
    let res = sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .expect("soft delete");
    assert_eq!(res.rows_affected(), 1, "soft delete should touch row {id}");
}

async fn restore(pool: &PgPool, id: &str) {
    sqlx::query("UPDATE knowledge_base SET deleted_at = NULL WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .expect("restore");
}

async fn write_entry(store: &PgKnowledgeBaseStore, id: &str, content: &str) {
    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new(id, content, vec!["notes".into()]))
            .await
            .unwrap_or_else(|e| panic!("write {id}: {e}"));
    })
    .await;
}

/// Boot a fixture with migrations applied, or pass-skip.
async fn fixture(name: &str) -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("kb656").await?;
    run_migrations(&fx.pool).await.expect("run_migrations");
    let _ = name;
    Some(fx)
}

fn embed_fn(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        calls.fetch_add(1, Ordering::SeqCst);
        let out: Vec<Vec<f32>> = texts.iter().map(|_| vec![0.1_f32, 0.2, 0.3]).collect();
        Box::pin(async move { Ok(out) })
    })
}

// --- read paths -------------------------------------------------------------

#[tokio::test]
async fn semantic_search_excludes_soft_deleted() {
    let Some(fx) = fixture("semantic_search_excludes_soft_deleted").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "live", "widget calibration notes").await;
    write_entry(&store, "retired", "widget calibration notes").await;
    set_embedding(&fx.pool, "live", vec![vec![1.0, 0.0, 0.0]]).await;
    set_embedding(&fx.pool, "retired", vec![vec![1.0, 0.0, 0.0]]).await;
    soft_delete(&fx.pool, "retired").await;

    // Query embedding points exactly at both rows, so only the deleted_at
    // predicate can separate them.
    let hits = with_user_id(UserId::new(USER), async {
        store
            .search("nomatchterm", vec![1.0, 0.0, 0.0], None, None, 10)
            .await
    })
    .await
    .expect("search");

    let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
    assert!(
        ids.contains(&"live"),
        "positive control: the live row must be reachable via the vector branch; got {ids:?}"
    );
    assert!(
        !ids.contains(&"retired"),
        "a soft-deleted entry must not surface from the vector branch; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn text_search_excludes_soft_deleted() {
    let Some(fx) = fixture("text_search_excludes_soft_deleted").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "live", "sprocket tolerance guidance").await;
    write_entry(&store, "retired", "sprocket tolerance guidance").await;
    soft_delete(&fx.pool, "retired").await;

    let hits = with_user_id(UserId::new(USER), async {
        store.search_text("sprocket", None, 10).await
    })
    .await
    .expect("search_text");

    let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
    assert!(ids.contains(&"live"), "positive control; got {ids:?}");
    assert!(
        !ids.contains(&"retired"),
        "a soft-deleted entry must not surface from full-text search; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn hybrid_search_excludes_soft_deleted_matched_by_both_branches() {
    let Some(fx) = fixture("hybrid_both_branches").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // The deleted row matches the FTS term AND sits at distance 0 from the
    // query vector, so it enters both CTEs. If either branch leaks it, the
    // FULL OUTER JOIN in `fused` surfaces it.
    write_entry(&store, "retired", "flywheel resonance dossier").await;
    set_embedding(&fx.pool, "retired", vec![vec![1.0, 0.0, 0.0]]).await;
    write_entry(&store, "live", "unrelated bookkeeping").await;
    set_embedding(&fx.pool, "live", vec![vec![0.0, 1.0, 0.0]]).await;
    soft_delete(&fx.pool, "retired").await;

    let hits = with_user_id(UserId::new(USER), async {
        store
            .search("flywheel", vec![1.0, 0.0, 0.0], None, None, 10)
            .await
    })
    .await
    .expect("search");

    let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
    assert!(
        !ids.contains(&"retired"),
        "a soft-deleted entry matched by BOTH branches must not survive fusion; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn soft_deleting_the_only_match_yields_empty_results() {
    let Some(fx) = fixture("only_match").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "retired", "singular quokka fact").await;
    set_embedding(&fx.pool, "retired", vec![vec![1.0, 0.0, 0.0]]).await;
    soft_delete(&fx.pool, "retired").await;

    // Boundary: an empty result set, not an error.
    let hits = with_user_id(UserId::new(USER), async {
        store
            .search("quokka", vec![1.0, 0.0, 0.0], None, None, 10)
            .await
    })
    .await
    .expect("search should succeed with no matches");
    assert!(hits.is_empty(), "expected no hits, got {hits:?}");
    fx.cleanup().await;
}

#[tokio::test]
async fn restored_entry_becomes_searchable_again() {
    let Some(fx) = fixture("restore").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "row", "restorable marmot fact").await;
    set_embedding(&fx.pool, "row", vec![vec![1.0, 0.0, 0.0]]).await;
    soft_delete(&fx.pool, "row").await;
    restore(&fx.pool, "row").await;

    let hits = with_user_id(UserId::new(USER), async {
        store
            .search("marmot", vec![1.0, 0.0, 0.0], None, None, 10)
            .await
    })
    .await
    .expect("search");
    assert!(
        hits.iter().any(|e| e.id == "row"),
        "clearing deleted_at must re-expose the row; got {hits:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn list_excludes_soft_deleted() {
    let Some(fx) = fixture("list").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "live", "kept").await;
    write_entry(&store, "retired", "pruned").await;
    soft_delete(&fx.pool, "retired").await;

    let entries = with_user_id(UserId::new(USER), async { store.list(50, 0, None).await })
        .await
        .expect("list");
    let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
    assert!(ids.contains(&"live"), "positive control; got {ids:?}");
    assert!(
        !ids.contains(&"retired"),
        "list must not include soft-deleted entries; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn list_page_excludes_soft_deleted() {
    let Some(fx) = fixture("list_page").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "live", "kept").await;
    write_entry(&store, "retired", "pruned").await;
    soft_delete(&fx.pool, "retired").await;

    let page = with_user_id(UserId::new(USER), async {
        store
            .list_page(KnowledgeListQuery {
                limit: 50,
                after: None,
                order: ListOrderOpt(ListOrder::NewestFirst),
                ..Default::default()
            })
            .await
    })
    .await
    .expect("list_page");

    let ids: Vec<&str> = page.entries.iter().map(|e| e.id.as_str()).collect();
    assert!(ids.contains(&"live"), "positive control; got {ids:?}");
    assert!(
        !ids.contains(&"retired"),
        "the KB browser must not page through soft-deleted entries; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn get_returns_none_for_soft_deleted() {
    let Some(fx) = fixture("get").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "retired", "pruned").await;
    soft_delete(&fx.pool, "retired").await;

    let got = with_user_id(UserId::new(USER), async { store.get("retired").await })
        .await
        .expect("get should succeed");
    assert!(
        got.is_none(),
        "fetching a retired entry by id must report absence, got {got:?}"
    );
    fx.cleanup().await;
}

// --- embedding pipeline -----------------------------------------------------

#[tokio::test]
async fn stale_invalidation_skips_soft_deleted() {
    let Some(fx) = fixture("invalidate").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    write_entry(&store, "retired", "pruned").await;
    set_embedding(&fx.pool, "retired", vec![vec![1.0, 0.0, 0.0]]).await;
    soft_delete(&fx.pool, "retired").await;

    // Model changed: 'model-A' -> 'model-B'. The tombstone is not searchable,
    // so clearing its vector is pure churn (on prod this re-embedded 681 rows).
    invalidate_stale_embeddings(&fx.pool, "model-B")
        .await
        .expect("invalidate");

    let still: Option<(bool,)> =
        sqlx::query_as("SELECT embedding IS NOT NULL FROM knowledge_base WHERE id = 'retired'")
            .fetch_optional(&fx.pool)
            .await
            .expect("probe");
    assert_eq!(
        still.map(|r| r.0),
        Some(true),
        "a soft-deleted row's embedding must be left alone by stale invalidation"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn backfill_skips_soft_deleted() {
    let Some(fx) = fixture("backfill").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // Both rows are unembedded and therefore backfill candidates; only the
    // live one should be picked up.
    write_entry(&store, "live", "kept").await;
    write_entry(&store, "retired", "pruned").await;
    soft_delete(&fx.pool, "retired").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = embed_fn(Arc::clone(&calls));
    let updated = backfill_knowledge_embeddings(
        &fx.pool,
        &embed,
        "model-A",
        &CancellationToken::new(),
    )
    .await
    .expect("backfill");

    assert_eq!(
        updated, 1,
        "backfill should embed only the live row, not the tombstone"
    );
    let retired_embedded: Option<(bool,)> =
        sqlx::query_as("SELECT embedding IS NOT NULL FROM knowledge_base WHERE id = 'retired'")
            .fetch_optional(&fx.pool)
            .await
            .expect("probe");
    assert_eq!(
        retired_embedded.map(|r| r.0),
        Some(false),
        "a soft-deleted row must not be embedded by the backfill"
    );
    fx.cleanup().await;
}
