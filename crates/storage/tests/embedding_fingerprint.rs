//! The embedding-model fingerprint is `<name>@<digest>`, and staleness must be
//! decided on the digest -- not on the whole string (issue #655).
//!
//! Renaming `nomic-embed-text:latest` to `nomic-embed-text` in `daemon.toml`
//! resolves to the *same* Ollama digest, but the whole-string comparison saw a
//! different fingerprint and invalidated everything. On `adele-prod` that
//! re-embedded 873 rows to produce vectors identical to the ones it replaced.
//! Cheap against a local Ollama; against a metered provider it is the whole
//! corpus re-embedded for a config typo, with search degraded until the
//! backfill drains.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test embedding_fingerprint
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_storage::embedding_backfill::invalidate_stale_embeddings;
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, run_migrations, with_user_id};
use pgvector::Vector;
use sqlx::PgPool;

const USER: &str = "kb-owner";
const DIGEST: &str = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("fp655").await?;
    run_migrations(&fx.pool).await.expect("run_migrations");
    Some(fx)
}

/// Write a knowledge row already embedded under `model`.
async fn embedded_row(pool: &PgPool, id: &str, model: &str) {
    let store = PgKnowledgeBaseStore::new(pool.clone());
    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new(id, "content", vec!["notes".into()]))
            .await
            .expect("write");
    })
    .await;
    let vecs: Vec<Vector> = vec![Vector::from(vec![0.1_f32, 0.2, 0.3])];
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = $2, embeddings_updated_at = NOW() \
         WHERE id = $3",
    )
    .bind(&vecs)
    .bind(model)
    .bind(id)
    .execute(pool)
    .await
    .expect("stamp embedding");
}

async fn stored(pool: &PgPool, id: &str) -> (bool, Option<String>) {
    sqlx::query_as::<_, (bool, Option<String>)>(
        "SELECT embedding IS NOT NULL, embedding_model FROM knowledge_base WHERE id = $1",
    )
    .bind(id)
    .fetch_one(pool)
    .await
    .expect("probe")
}

#[tokio::test]
async fn same_digest_different_name_does_not_invalidate() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    embedded_row(
        &fx.pool,
        "row",
        &format!("nomic-embed-text:latest@{DIGEST}"),
    )
    .await;

    invalidate_stale_embeddings(&fx.pool, &format!("nomic-embed-text@{DIGEST}"))
        .await
        .expect("invalidate");

    let (has_embedding, _) = stored(&fx.pool, "row").await;
    assert!(
        has_embedding,
        "the model is byte-identical; renaming it in config must not discard the vector"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn same_digest_different_name_restamps_stored_fingerprint() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let current = format!("nomic-embed-text@{DIGEST}");
    embedded_row(
        &fx.pool,
        "row",
        &format!("nomic-embed-text:latest@{DIGEST}"),
    )
    .await;

    invalidate_stale_embeddings(&fx.pool, &current)
        .await
        .expect("invalidate");

    let (_, model) = stored(&fx.pool, "row").await;
    assert_eq!(
        model.as_deref(),
        Some(current.as_str()),
        "the row must adopt the new spelling so the check converges instead of \
         re-evaluating every boot"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn different_digest_invalidates() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    embedded_row(&fx.pool, "row", &format!("nomic-embed-text@{DIGEST}")).await;

    // A genuinely different model must still drop its vectors.
    invalidate_stale_embeddings(&fx.pool, "mxbai-embed-large@ffff0000")
        .await
        .expect("invalidate");

    let (has_embedding, model) = stored(&fx.pool, "row").await;
    assert!(!has_embedding, "a real model change must invalidate");
    assert_eq!(
        model, None,
        "and clear the stamp so the backfill picks it up"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn missing_digest_on_the_stored_side_falls_back_to_string_compare() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    // Older rows were stamped with a bare name. With nothing to compare, stay
    // conservative and invalidate rather than assume they match.
    embedded_row(&fx.pool, "row", "nomic-embed-text").await;

    invalidate_stale_embeddings(&fx.pool, &format!("nomic-embed-text@{DIGEST}"))
        .await
        .expect("invalidate");

    let (has_embedding, _) = stored(&fx.pool, "row").await;
    assert!(
        !has_embedding,
        "no digest on the stored side means no proof of sameness; invalidate"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn missing_digest_on_the_current_side_falls_back_to_string_compare() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    // The connector could not resolve a digest this boot (it falls back to the
    // configured name). Same reasoning: no proof, so invalidate.
    embedded_row(&fx.pool, "row", &format!("nomic-embed-text@{DIGEST}")).await;

    invalidate_stale_embeddings(&fx.pool, "nomic-embed-text")
        .await
        .expect("invalidate");

    let (has_embedding, _) = stored(&fx.pool, "row").await;
    assert!(!has_embedding, "no digest to compare; invalidate");
    fx.cleanup().await;
}

#[tokio::test]
async fn identical_fingerprint_is_untouched() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let current = format!("nomic-embed-text@{DIGEST}");
    embedded_row(&fx.pool, "row", &current).await;

    let (kb, _) = invalidate_stale_embeddings(&fx.pool, &current)
        .await
        .expect("invalidate");

    assert_eq!(kb, 0, "the happy path must report no work done");
    let (has_embedding, model) = stored(&fx.pool, "row").await;
    assert!(has_embedding);
    assert_eq!(model.as_deref(), Some(current.as_str()));
    fx.cleanup().await;
}

#[tokio::test]
async fn tool_definitions_same_digest_is_restamped_not_invalidated() {
    let Some(fx) = fixture().await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let current = format!("nomic-embed-text@{DIGEST}");
    let vecs: Vec<Vector> = vec![Vector::from(vec![0.1_f32, 0.2, 0.3])];
    sqlx::query(
        "INSERT INTO tool_definitions (name, description, parameters, source, embedding, embedding_model) \
         VALUES ('t', 'd', '{}'::jsonb, 'test', $1::vector[], $2)",
    )
    .bind(&vecs)
    .bind(format!("nomic-embed-text:latest@{DIGEST}"))
    .execute(&fx.pool)
    .await
    .expect("seed tool");

    invalidate_stale_embeddings(&fx.pool, &current)
        .await
        .expect("invalidate");

    let (has_embedding, model) = sqlx::query_as::<_, (bool, Option<String>)>(
        "SELECT embedding IS NOT NULL, embedding_model FROM tool_definitions WHERE name = 't'",
    )
    .fetch_one(&fx.pool)
    .await
    .expect("probe");
    assert!(has_embedding, "tool embeddings get the same treatment");
    assert_eq!(model.as_deref(), Some(current.as_str()));
    fx.cleanup().await;
}
