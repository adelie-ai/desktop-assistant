//! Background backfill of missing or stale embeddings.
//!
//! Selects rows where the embedding is NULL, the model stamp is NULL, or the
//! model stamp doesn't match the current model, then generates and writes the
//! embedding in batches.  Naturally idempotent — incomplete runs resume on
//! next startup.

use std::future::Future;
use std::pin::Pin;

use desktop_assistant_core::chunking::{chunk_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP};
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
/// Also cleans up orphaned state where `embedding_model` is set but `embedding`
/// is NULL (e.g. from a previous interrupted invalidation or failed backfill).
///
/// Returns `(knowledge_count, tool_count)` of invalidated rows.
pub async fn invalidate_stale_embeddings(
    pool: &PgPool,
    current_model: &str,
) -> Result<(u64, u64), String> {
    // Invalidate stale model embeddings (model mismatch).
    let kb_stale = sqlx::query(
        "UPDATE knowledge_base
         SET embedding = NULL, embedding_model = NULL
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model != $1",
    )
    .bind(current_model)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    // Clean up orphaned state: model is set but embedding is NULL.
    let kb_orphan = sqlx::query(
        "UPDATE knowledge_base
         SET embedding_model = NULL
         WHERE embedding IS NULL
           AND embedding_model IS NOT NULL",
    )
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let tool_stale = sqlx::query(
        "UPDATE tool_definitions
         SET embedding = NULL, embedding_model = NULL
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model != $1",
    )
    .bind(current_model)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let tool_orphan = sqlx::query(
        "UPDATE tool_definitions
         SET embedding_model = NULL
         WHERE embedding IS NULL
           AND embedding_model IS NOT NULL",
    )
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let kb_total = kb_stale.rows_affected() + kb_orphan.rows_affected();
    let tool_total = tool_stale.rows_affected() + tool_orphan.rows_affected();
    Ok((kb_total, tool_total))
}

/// Backfill embeddings for `knowledge_base` rows that are missing or stale.
///
/// Each entry's content is split into chunks, all chunks are batch-embedded,
/// and the resulting vectors are stored as a `vector[]` array on the row.
///
/// Continues past batch failures so that a single bad batch does not block the
/// entire backfill.  Returns the total number of rows successfully updated.
pub async fn backfill_knowledge_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;
    let mut consecutive_failures = 0u32;

    loop {
        // Select rows needing embedding.  Rows where embedding_model already
        // matches the current model are either already done (embedding present)
        // or failed individually on a prior iteration (embedding NULL, model
        // stamped) — skip both to avoid an infinite retry loop.
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT id, content FROM knowledge_base
             WHERE embedding_model IS NULL
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

        // Chunk all rows and track which chunks belong to which row.
        let mut all_chunks: Vec<(usize, String)> = Vec::new();
        for (i, (_, content)) in rows.iter().enumerate() {
            for chunk in chunk_text(content, CHUNK_MAX_CHARS, CHUNK_OVERLAP) {
                all_chunks.push((i, chunk));
            }
        }

        let texts: Vec<String> = all_chunks.iter().map(|(_, t)| t.clone()).collect();
        match embed_fn(texts).await {
            Ok(embeddings) => {
                consecutive_failures = 0;
                // Group embeddings back by row index.
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings.into_iter()) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((id, _), vecs) in rows.iter().zip(row_embeddings.into_iter()) {
                    sqlx::query(
                        "UPDATE knowledge_base
                         SET embedding = $1::vector[], embedding_model = $2
                         WHERE id = $3",
                    )
                    .bind(&vecs)
                    .bind(current_model)
                    .bind(id)
                    .execute(pool)
                    .await
                    .map_err(|e| e.to_string())?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("knowledge embedding batch failed, retrying individually: {e}");
                // Batch failed — retry each entry individually so good entries still get embedded.
                let mut any_succeeded = false;
                for (id, content) in &rows {
                    let chunks = chunk_text(content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    match embed_fn(chunks).await {
                        Ok(embeddings) => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            sqlx::query(
                                "UPDATE knowledge_base
                                 SET embedding = $1::vector[], embedding_model = $2
                                 WHERE id = $3",
                            )
                            .bind(&vecs)
                            .bind(current_model)
                            .bind(id)
                            .execute(pool)
                            .await
                            .map_err(|e| e.to_string())?;
                            total += 1;
                            any_succeeded = true;
                        }
                        Err(e2) => {
                            tracing::warn!("skipping knowledge entry {id}: {e2}");
                            // Mark it so we don't retry it every startup.
                            sqlx::query(
                                "UPDATE knowledge_base
                                 SET embedding_model = $1
                                 WHERE id = $2",
                            )
                            .bind(current_model)
                            .bind(id)
                            .execute(pool)
                            .await
                            .map_err(|e| e.to_string())?;
                        }
                    }
                }
                if any_succeeded {
                    consecutive_failures = 0;
                } else {
                    consecutive_failures += 1;
                    if consecutive_failures >= 3 {
                        tracing::error!(
                            "knowledge embedding backfill aborting after {consecutive_failures} consecutive failures"
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(total)
}

/// Backfill embeddings for `tool_definitions` rows that are missing or stale.
///
/// The text embedded is `name || ' ' || description` to match the tsvector.
///
/// Continues past batch failures so that a single bad batch does not block the
/// entire backfill.  Returns the total number of rows successfully updated.
pub async fn backfill_tool_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;
    let mut consecutive_failures = 0u32;

    loop {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT name, name || ' ' || description AS text
             FROM tool_definitions
             WHERE embedding_model IS NULL
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
        match embed_fn(texts).await {
            Ok(embeddings) => {
                consecutive_failures = 0;
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
            Err(e) => {
                tracing::warn!("tool embedding batch failed, retrying individually: {e}");
                let mut any_succeeded = false;
                for (name, text) in &rows {
                    match embed_fn(vec![text.clone()]).await {
                        Ok(embeddings) => {
                            if let Some(embedding) = embeddings.into_iter().next() {
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
                                total += 1;
                                any_succeeded = true;
                            }
                        }
                        Err(e2) => {
                            tracing::warn!("skipping tool {name}: {e2}");
                            sqlx::query(
                                "UPDATE tool_definitions
                                 SET embedding_model = $1
                                 WHERE name = $2",
                            )
                            .bind(current_model)
                            .bind(name)
                            .execute(pool)
                            .await
                            .map_err(|e| e.to_string())?;
                        }
                    }
                }
                if any_succeeded {
                    consecutive_failures = 0;
                } else {
                    consecutive_failures += 1;
                    if consecutive_failures >= 3 {
                        tracing::error!(
                            "tool embedding backfill aborting after {consecutive_failures} consecutive failures"
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(total)
}
