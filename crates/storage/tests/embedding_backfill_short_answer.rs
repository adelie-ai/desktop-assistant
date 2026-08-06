//! DB-touching integration tests for issue #1108.
//!
//! `backfill_knowledge_embeddings`, `backfill_tool_embeddings` and
//! `backfill_skill_embeddings` used to write whatever the embedder returned
//! without checking it against the number of chunks it was handed. A provider
//! that silently caps its batch size then leaves a row holding a truncated
//! `vector[]` -- covering only the first N chunks -- stamped as current. The
//! stamp hides the loss from both the operator and the next backfill pass.
//!
//! These pin the fix: a length mismatch, at either the batch call or the
//! per-row retry, clears the vector instead of writing a partial one, stamps
//! the row so the pass still converges, and logs both counts.
//!
//! `TEST_DATABASE_URL` gates every test the same way as
//! `embedding_backfill_retry.rs`; see that file's header for how to run
//! locally (`just test-db`).

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use desktop_assistant_storage::embedding_backfill::{
    BackfillEmbedFn, backfill_knowledge_embeddings, backfill_skill_embeddings,
    backfill_tool_embeddings,
};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Fake embedders
// ---------------------------------------------------------------------------

fn ok_vecs(n: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| vec![0.1_f32, 0.2, 0.3]).collect()
}

/// Always answers with one fewer vector than the number of texts it was
/// handed -- models a provider that silently caps its batch size on every
/// call, batch and per-row retry alike.
fn short_by_one_embed(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        calls.fetch_add(1, Ordering::SeqCst);
        let n = texts.len().saturating_sub(1);
        Box::pin(async move { Ok(ok_vecs(n)) })
    })
}

/// Answers short only on its first call (the batch call); every later call
/// (the per-row retry) answers with the correct count. Models a provider
/// whose cap bites a multi-item batch but not a single-item request, so the
/// pass recovers without losing anything.
fn short_batch_then_correct_embed(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        let call_no = calls.fetch_add(1, Ordering::SeqCst);
        let n = if call_no == 0 {
            texts.len().saturating_sub(1)
        } else {
            texts.len()
        };
        Box::pin(async move { Ok(ok_vecs(n)) })
    })
}

/// Fails its first call (the batch call) with a plain backend error, then
/// answers short on every later call (the per-row retry). Isolates the
/// per-row retry's own mismatch message from the batch path's -- the batch
/// call here never produces a count-mismatch message of its own.
fn err_on_batch_then_short_individually(calls: Arc<AtomicUsize>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        let call_no = calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if call_no == 0 {
                Err("simulated batch failure".to_string())
            } else {
                Ok(ok_vecs(texts.len().saturating_sub(1)))
            }
        })
    })
}

/// Content long enough that `chunk_text` splits it into more than one chunk
/// (`long_text_gets_chunked` in `core::chunking` pins the same repeat, at
/// ~2500 chars, against `CHUNK_MAX_CHARS = 800`).
fn multi_chunk_content() -> String {
    "word ".repeat(500)
}

// ---------------------------------------------------------------------------
// knowledge_base seed / read helpers
// ---------------------------------------------------------------------------

async fn seed_knowledge(pool: &PgPool, id: &str, content: &str) {
    sqlx::query("INSERT INTO knowledge_base (id, user_id, content) VALUES ($1, 'default', $2)")
        .bind(id)
        .bind(content)
        .execute(pool)
        .await
        .expect("seed knowledge_base row");
}

async fn knowledge_embedding_is_set(pool: &PgPool, id: &str) -> bool {
    sqlx::query_scalar("SELECT embedding IS NOT NULL FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read embedding")
}

async fn knowledge_embedding_model(pool: &PgPool, id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT embedding_model FROM knowledge_base WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("read embedding_model")
}

// ---------------------------------------------------------------------------
// tool_definitions seed / read helpers
// ---------------------------------------------------------------------------

async fn seed_tool(pool: &PgPool, name: &str, description: &str) {
    sqlx::query(
        "INSERT INTO tool_definitions (name, description, parameters, source, is_core)
         VALUES ($1, $2, '{}'::jsonb, 'test', false)",
    )
    .bind(name)
    .bind(description)
    .execute(pool)
    .await
    .expect("seed tool_definitions row");
}

async fn tool_embedding_is_set(pool: &PgPool, name: &str) -> bool {
    sqlx::query_scalar("SELECT embedding IS NOT NULL FROM tool_definitions WHERE name = $1")
        .bind(name)
        .fetch_one(pool)
        .await
        .expect("read embedding")
}

async fn tool_embedding_model(pool: &PgPool, name: &str) -> Option<String> {
    sqlx::query_scalar("SELECT embedding_model FROM tool_definitions WHERE name = $1")
        .bind(name)
        .fetch_one(pool)
        .await
        .expect("read embedding_model")
}

// ---------------------------------------------------------------------------
// skill_index seed / read helpers
// ---------------------------------------------------------------------------

async fn seed_skill(pool: &PgPool, name: &str, description: &str) {
    sqlx::query(
        "INSERT INTO skill_index (name, description, disk_path, content_hash)
         VALUES ($1, $2, '/dev/null', 'deadbeef')",
    )
    .bind(name)
    .bind(description)
    .execute(pool)
    .await
    .expect("seed skill_index row");
}

async fn skill_embedding_is_set(pool: &PgPool, name: &str) -> bool {
    sqlx::query_scalar(
        "SELECT embedding IS NOT NULL FROM skill_index WHERE name = $1 AND owner_key = ''",
    )
    .bind(name)
    .fetch_one(pool)
    .await
    .expect("read embedding")
}

async fn skill_embedding_model(pool: &PgPool, name: &str) -> Option<String> {
    sqlx::query_scalar("SELECT embedding_model FROM skill_index WHERE name = $1 AND owner_key = ''")
        .bind(name)
        .fetch_one(pool)
        .await
        .expect("read embedding_model")
}

// ---------------------------------------------------------------------------
// knowledge_base tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn knowledge_multi_chunk_row_is_cleared_not_truncated_when_the_embedder_answers_short() {
    let Some(fx) = support::DbFixture::try_new("embed1108kb").await else {
        return;
    };
    let pool = &fx.pool;
    seed_knowledge(pool, "kb-1", &multi_chunk_content()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_by_one_embed(calls.clone());
    let total = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill returns Ok even when every row fails");

    assert_eq!(total, 0, "a short answer is not a success");
    assert!(
        !knowledge_embedding_is_set(pool, "kb-1").await,
        "a short answer must clear the vector, not store the first N chunks"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_short_answer_row_is_stamped_so_the_backfill_converges() {
    let Some(fx) = support::DbFixture::try_new("embed1108kb").await else {
        return;
    };
    let pool = &fx.pool;
    seed_knowledge(pool, "kb-1", "short fact").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_by_one_embed(calls.clone());
    let first = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill returns Ok");
    assert_eq!(first, 0);
    assert_eq!(
        knowledge_embedding_model(pool, "kb-1").await.as_deref(),
        Some("model-A"),
        "the row must be stamped even though it holds no vector"
    );
    let calls_after_first = calls.load(Ordering::SeqCst);
    assert!(calls_after_first > 0, "the embedder ran on the first pass");

    let second = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("second backfill succeeds");
    assert_eq!(second, 0, "the stamped row is not re-selected");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        calls_after_first,
        "no further embedder calls on the second pass"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_batch_short_answer_falls_back_to_individual_retry_and_recovers() {
    let Some(fx) = support::DbFixture::try_new("embed1108kb").await else {
        return;
    };
    let pool = &fx.pool;
    seed_knowledge(pool, "kb-1", "fact one").await;
    seed_knowledge(pool, "kb-2", "fact two").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_batch_then_correct_embed(calls.clone());
    let total = backfill_knowledge_embeddings(pool, &embed, "model-A", &CancellationToken::new())
        .await
        .expect("backfill succeeds");

    assert_eq!(
        total, 2,
        "the batch guard must fall back to the per-row retry instead of \
         writing the mismatched batch"
    );
    assert!(knowledge_embedding_is_set(pool, "kb-1").await);
    assert!(knowledge_embedding_is_set(pool, "kb-2").await);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "one rejected batch call plus one retry call per row"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_batch_mismatch_log_line_names_returned_and_expected_counts() {
    let Some(fx) = support::DbFixture::try_new("embed1108kb").await else {
        return;
    };
    let pool = &fx.pool;
    seed_knowledge(pool, "kb-1", "fact one").await;
    seed_knowledge(pool, "kb-2", "fact two").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_batch_then_correct_embed(calls.clone());
    let token = CancellationToken::new();
    let (_, log) =
        support::capture_tracing(|| backfill_knowledge_embeddings(pool, &embed, "model-A", &token))
            .await;

    assert!(
        log.contains("returned 1 vector(s) for 2 chunk(s)"),
        "log line must name both the returned and the expected count; got: {log}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_per_row_mismatch_log_line_names_returned_and_expected_counts() {
    let Some(fx) = support::DbFixture::try_new("embed1108kb").await else {
        return;
    };
    let pool = &fx.pool;
    seed_knowledge(pool, "kb-1", "fact one").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = err_on_batch_then_short_individually(calls.clone());
    let token = CancellationToken::new();
    let (_, log) =
        support::capture_tracing(|| backfill_knowledge_embeddings(pool, &embed, "model-A", &token))
            .await;

    assert!(
        log.contains("returned 0 vector(s) for 1 chunk(s)"),
        "log line must name both the returned and the expected count; got: {log}"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// tool_definitions tests (#1108: identical shape to knowledge_base)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_definitions_multi_chunk_row_is_cleared_not_truncated_when_the_embedder_answers_short()
{
    let Some(fx) = support::DbFixture::try_new("embed1108tool").await else {
        return;
    };
    let pool = &fx.pool;
    seed_tool(pool, "big-tool", &multi_chunk_content()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_by_one_embed(calls.clone());
    let total = backfill_tool_embeddings(pool, &embed, "model-A")
        .await
        .expect("backfill returns Ok even when every row fails");

    assert_eq!(total, 0, "a short answer is not a success");
    assert!(
        !tool_embedding_is_set(pool, "big-tool").await,
        "a short answer must clear the vector, not store the first N chunks"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn tool_definitions_short_answer_row_is_stamped_so_the_backfill_converges() {
    let Some(fx) = support::DbFixture::try_new("embed1108tool").await else {
        return;
    };
    let pool = &fx.pool;
    seed_tool(pool, "small-tool", "does a thing").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_by_one_embed(calls.clone());
    let first = backfill_tool_embeddings(pool, &embed, "model-A")
        .await
        .expect("backfill returns Ok");
    assert_eq!(first, 0);
    assert_eq!(
        tool_embedding_model(pool, "small-tool").await.as_deref(),
        Some("model-A"),
        "the row must be stamped even though it holds no vector"
    );
    let calls_after_first = calls.load(Ordering::SeqCst);
    assert!(calls_after_first > 0, "the embedder ran on the first pass");

    let second = backfill_tool_embeddings(pool, &embed, "model-A")
        .await
        .expect("second backfill succeeds");
    assert_eq!(second, 0, "the stamped row is not re-selected");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        calls_after_first,
        "no further embedder calls on the second pass"
    );

    fx.cleanup().await;
}

// ---------------------------------------------------------------------------
// skill_index tests (#1108: identical shape to knowledge_base)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn skill_index_multi_chunk_row_is_cleared_not_truncated_when_the_embedder_answers_short() {
    let Some(fx) = support::DbFixture::try_new("embed1108skill").await else {
        return;
    };
    let pool = &fx.pool;
    seed_skill(pool, "big-skill", &multi_chunk_content()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_by_one_embed(calls.clone());
    let total = backfill_skill_embeddings(pool, &embed, "model-A")
        .await
        .expect("backfill returns Ok even when every row fails");

    assert_eq!(total, 0, "a short answer is not a success");
    assert!(
        !skill_embedding_is_set(pool, "big-skill").await,
        "a short answer must clear the vector, not store the first N chunks"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn skill_index_short_answer_row_is_stamped_so_the_backfill_converges() {
    let Some(fx) = support::DbFixture::try_new("embed1108skill").await else {
        return;
    };
    let pool = &fx.pool;
    seed_skill(pool, "small-skill", "does a thing").await;

    let calls = Arc::new(AtomicUsize::new(0));
    let embed = short_by_one_embed(calls.clone());
    let first = backfill_skill_embeddings(pool, &embed, "model-A")
        .await
        .expect("backfill returns Ok");
    assert_eq!(first, 0);
    assert_eq!(
        skill_embedding_model(pool, "small-skill").await.as_deref(),
        Some("model-A"),
        "the row must be stamped even though it holds no vector"
    );
    let calls_after_first = calls.load(Ordering::SeqCst);
    assert!(calls_after_first > 0, "the embedder ran on the first pass");

    let second = backfill_skill_embeddings(pool, &embed, "model-A")
        .await
        .expect("second backfill succeeds");
    assert_eq!(second, 0, "the stamped row is not re-selected");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        calls_after_first,
        "no further embedder calls on the second pass"
    );

    fx.cleanup().await;
}
