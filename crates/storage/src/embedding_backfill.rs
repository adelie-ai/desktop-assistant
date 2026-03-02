//! Background backfill of missing or stale embeddings.
//!
//! Selects rows where the embedding is NULL, the model stamp is NULL, or the
//! model stamp doesn't match the current model, then generates and writes the
//! embedding in batches.  Naturally idempotent — incomplete runs resume on
//! next startup.

use std::future::Future;
use std::pin::Pin;

use pgvector::Vector;
use sqlx::PgPool;

/// Boxed async embedding function: takes a list of texts, returns a list of vectors.
pub type BackfillEmbedFn = Box<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, String>> + Send>>
        + Send
        + Sync,
>;

const BATCH_SIZE: i64 = 32;

/// Invalidate (NULL-out) embeddings whose model stamp doesn't match the current model.
///
/// This prevents pgvector dimension-mismatch errors when the embedding model
/// changes (e.g. switching from a 1536-dim to 768-dim model).  Rows with
/// NULL embeddings are silently skipped by vector search and will be re-embedded
/// by the backfill loop.
///
/// Returns `(knowledge_count, tool_count)` of invalidated rows.
pub async fn invalidate_stale_embeddings(
    pool: &PgPool,
    current_model: &str,
) -> Result<(u64, u64), String> {
    let kb_result = sqlx::query(
        "UPDATE knowledge_base
         SET embedding = NULL
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model != $1",
    )
    .bind(current_model)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let tool_result = sqlx::query(
        "UPDATE tool_definitions
         SET embedding = NULL
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model != $1",
    )
    .bind(current_model)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok((kb_result.rows_affected(), tool_result.rows_affected()))
}

/// Backfill embeddings for `knowledge_base` rows that are missing or stale.
///
/// Returns the total number of rows updated.
pub async fn backfill_knowledge_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;

    loop {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT id, content FROM knowledge_base
             WHERE embedding IS NULL
                OR embedding_model IS NULL
                OR embedding_model != $1
             LIMIT $2",
        )
        .bind(current_model)
        .bind(BATCH_SIZE)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

        if rows.is_empty() {
            break;
        }

        let texts: Vec<String> = rows.iter().map(|(_, content)| content.clone()).collect();
        let embeddings = embed_fn(texts).await?;

        for ((id, _), embedding) in rows.iter().zip(embeddings.into_iter()) {
            let vec = Vector::from(embedding);
            sqlx::query(
                "UPDATE knowledge_base
                 SET embedding = $1, embedding_model = $2
                 WHERE id = $3",
            )
            .bind(vec)
            .bind(current_model)
            .bind(id)
            .execute(pool)
            .await
            .map_err(|e| e.to_string())?;
        }

        total += rows.len();
    }

    Ok(total)
}

/// Backfill embeddings for `tool_definitions` rows that are missing or stale.
///
/// The text embedded is `name || ' ' || description` to match the tsvector.
///
/// Returns the total number of rows updated.
pub async fn backfill_tool_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;

    loop {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT name, name || ' ' || description AS text
             FROM tool_definitions
             WHERE embedding IS NULL
                OR embedding_model IS NULL
                OR embedding_model != $1
             LIMIT $2",
        )
        .bind(current_model)
        .bind(BATCH_SIZE)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

        if rows.is_empty() {
            break;
        }

        let texts: Vec<String> = rows.iter().map(|(_, text)| text.clone()).collect();
        let embeddings = embed_fn(texts).await?;

        for ((name, _), embedding) in rows.iter().zip(embeddings.into_iter()) {
            let vec = Vector::from(embedding);
            sqlx::query(
                "UPDATE tool_definitions
                 SET embedding = $1, embedding_model = $2
                 WHERE name = $3",
            )
            .bind(vec)
            .bind(current_model)
            .bind(name)
            .execute(pool)
            .await
            .map_err(|e| e.to_string())?;
        }

        total += rows.len();
    }

    Ok(total)
}
