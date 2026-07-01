//! DB-touching integration tests for `embedding_backfill` (issue #436).
//!
//! `backfill_knowledge_embeddings` is the resilience the dream cycle depends
//! on when the embedder wedges: a batch failure falls back to per-row retry, a
//! persistently-failing row is *stamped* so it is not re-looped, and the pass
//! aborts after too many consecutive failures. Before this suite that logic had
//! zero test references. These pin it with fake `embed_fn` closures
//! (always-Ok / always-Err / batch-fail-then-individual-Ok / cancel) so no real
//! embedder is needed.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! psql "$URL" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test embedding_backfill_retry
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use desktop_assistant_storage::embedding_backfill::{
    BackfillEmbedFn, backfill_knowledge_embeddings,
};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Fake embedders
// ---------------------------------------------------------------------------

/// One small vector per input text. The `knowledge_base.embedding` column is an
/// unmodified `vector[]`, so a fixed 3-dim vector stores fine.
fn ok_vecs(n: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| vec![0.1_f32, 0.2, 0.3]).collect()
}

/// Always succeeds; bumps `calls` once per invocation.
fn always_ok_embed(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        calls.fetch_add(1, Ordering::SeqCst);
        let out = ok_vecs(texts.len());
        Box::pin(async move { Ok(out) })
    })
}

/// Fails when handed more than one text (the batch call), succeeds for a
/// single-text call (the per-row retry). Models a wedged batch endpoint whose
/// single-item requests still go through.
fn batch_fail_individual_ok(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        calls.fetch_add(1, Ordering::SeqCst);
        let len = texts.len();
        Box::pin(async move {
            if len > 1 {
                Err("simulated batch embedding failure".to_string())
            } else {
                Ok(ok_vecs(len))
            }
        })
    })
}

/// Always fails; bumps `calls` once per invocation.
fn always_err_embed(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |_texts: Vec<String>| {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Err("simulated persistent embedding failure".to_string()) })
    })
}

/// Cancels `token` on its first call, then succeeds — so the first batch lands
/// but the between-batches check must stop the loop.
fn ok_and_cancel(token: CancellationToken) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        token.cancel();
        let out = ok_vecs(texts.len());
        Box::pin(async move { Ok(out) })
    })
}

// ---------------------------------------------------------------------------
// Seed / read helpers
// ---------------------------------------------------------------------------

/// Seed a KB row with a NULL embedding and no model stamp — i.e. one the
/// backfill's selection predicate will pick up.
async fn seed_unembedded(pool: &PgPool, id: &str, content: &str) {
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content) VALUES ($1, 'default', $2)")
        .bind(id)
        .bind(content)
        .execute(pool)
        .await
        .expect("seed unembedded knowledge_base row");
}

async fn embedding_is_set(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT embedding IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read embedding")
}

async fn embedding_model(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT embedding_model FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read embedding_model")
}

async fn embeddings_updated_at_is_set(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT embeddings_updated_at IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read embeddings_updated_at")
}

async fn count_stamped(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM knowledge_base WHERE embedding_model IS NOT NULL")
        .fetch_one(pool)
        .await
        .expect("count stamped rows")
}

async fn count_unstamped(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM knowledge_base WHERE embedding_model IS NULL")
        .fetch_one(pool)
        .await
        .expect("count unstamped rows")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn backfill_writes_vectors_and_stamps_model() {
    let Some(fx) = support::DbFixture::try_new("embed436").await else {
        return;
    };
    let pool = &fx.pool;
    seed_unembedded(pool, "kb-1", "hello world").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = always_ok_embed(calls.clone());
    let total = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill succeeds");

    assert_eq!(total, 1, "one row embedded");
    // Central assertion: the vector is written and the row stamped with the model
    // + freshness marker. Dropping the model from the success UPDATE breaks this.
    assert!(
        embedding_is_set(pool, "kb-1").await,
        "embedding vector written"
    );
    assert_eq!(
        embedding_model(pool, "kb-1").await.as_deref(),
        Some("model-A"),
        "model stamped"
    );
    assert!(
        embeddings_updated_at_is_set(pool, "kb-1").await,
        "freshness stamp written"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn backfill_retries_rows_individually_after_batch_failure() {
    let Some(fx) = support::DbFixture::try_new("embed436").await else {
        return;
    };
    let pool = &fx.pool;
    seed_unembedded(pool, "kb-1", "first short fact").await;
    seed_unembedded(pool, "kb-2", "second short fact").await;

    // The batch (both rows' single chunks) fails; each per-row retry succeeds.
    let calls = Arc::new(AtomicUsize::new(0));
    let embed = batch_fail_individual_ok(calls.clone());
    let total = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill succeeds");

    // Central assertion: despite the batch failure, both rows are embedded via
    // the per-row retry path.
    assert_eq!(total, 2, "both rows embedded via individual retry");
    assert!(embedding_is_set(pool, "kb-1").await);
    assert!(embedding_is_set(pool, "kb-2").await);
    assert_eq!(
        embedding_model(pool, "kb-1").await.as_deref(),
        Some("model-A")
    );
    assert_eq!(
        embedding_model(pool, "kb-2").await.as_deref(),
        Some("model-A")
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn backfill_stamps_failed_rows_so_they_are_not_reselected() {
    let Some(fx) = support::DbFixture::try_new("embed436").await else {
        return;
    };
    let pool = &fx.pool;
    seed_unembedded(pool, "kb-1", "fact one").await;
    seed_unembedded(pool, "kb-2", "fact two").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = always_err_embed(calls.clone());
    let first = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill returns Ok even when every row fails");
    assert_eq!(first, 0, "nothing successfully embedded");

    // Each failed row keeps a NULL vector but IS stamped (model + freshness), so
    // it drops out of the selection predicate.
    for id in ["kb-1", "kb-2"] {
        assert!(
            !embedding_is_set(pool, id).await,
            "vector still NULL after a failed embed"
        );
        assert_eq!(
            embedding_model(pool, id).await.as_deref(),
            Some("model-A"),
            "failed row stamped with the model so it is not re-selected"
        );
        assert!(
            embeddings_updated_at_is_set(pool, id).await,
            "failed row stamped with a freshness marker"
        );
    }

    let calls_after_first = calls.load(Ordering::SeqCst);
    assert!(calls_after_first > 0, "the embedder ran on the first pass");

    // Central assertion: a second run selects nothing — the stamped rows are not
    // re-looped, so the embedder is never called again. Removing the failed-row
    // stamp lets them be re-selected and re-embedded here.
    let second = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("second backfill succeeds");
    assert_eq!(second, 0, "second run selects and embeds nothing");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        calls_after_first,
        "no further embedder calls on the second run"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn backfill_aborts_after_three_consecutive_failures() {
    let Some(fx) = support::DbFixture::try_new("embed436").await else {
        return;
    };
    let pool = &fx.pool;
    // 100 rows → four selection batches of 32/32/32/4 (BATCH_SIZE = 32). With an
    // always-failing embedder each batch is a consecutive failure; the loop must
    // abort once the count reaches 3 — before the 4th batch is ever selected.
    for i in 0..100 {
        seed_unembedded(pool, &format!("kb-{i:03}"), "fact").await;
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = always_err_embed(calls.clone());
    let total = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill returns Ok after aborting");
    assert_eq!(total, 0, "no rows embedded");

    // Central assertion: exactly 3 batches (96 rows) were processed before the
    // abort, leaving the 4th batch's rows untouched. Raising the abort threshold
    // stamps all 100.
    assert_eq!(
        count_stamped(pool).await,
        96,
        "3 batches (96 rows) processed before abort"
    );
    assert_eq!(
        count_unstamped(pool).await,
        4,
        "the 4th batch was never reached"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn backfill_honors_cancellation_between_batches() {
    let Some(fx) = support::DbFixture::try_new("embed436").await else {
        return;
    };
    let pool = &fx.pool;
    // 40 rows → batch 1 (32) + batch 2 (8) under BATCH_SIZE = 32.
    for i in 0..40 {
        seed_unembedded(pool, &format!("kb-{i:03}"), "fact").await;
    }

    // The embedder cancels the token during the first batch, then returns Ok — so
    // batch 1 completes but the between-batches check must skip batch 2.
    let token = CancellationToken::new();
    let embed = ok_and_cancel(token.clone());
    let total = backfill_knowledge_embeddings(pool, &embed, "model-A", &token)
        .await
        .expect("backfill succeeds");

    // Central assertion: only the first batch (32 rows) is embedded; the second
    // batch is skipped by the between-batches cancellation check.
    assert_eq!(
        total, 32,
        "only the first batch was processed before cancellation"
    );
    assert_eq!(count_stamped(pool).await, 32, "first batch embedded");
    assert_eq!(
        count_unstamped(pool).await,
        8,
        "second batch skipped by cancellation"
    );

    fx.cleanup().await;
}
