//! Background backfill of missing or stale embeddings.
//!
//! Selects rows where the embedding is NULL, the model stamp is NULL, or the
//! model stamp doesn't match the current model, then generates and writes the
//! embedding in batches.  Naturally idempotent — incomplete runs resume on
//! next startup.

use std::future::Future;
use std::pin::Pin;

use desktop_assistant_core::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};
use pgvector::Vector;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

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
///
/// # Staleness is decided on the digest
///
/// The stamp is a fingerprint of the form `<name>@<digest>`. Comparing it as a
/// whole string makes a purely cosmetic rename (`nomic-embed-text:latest` ->
/// `nomic-embed-text`, same model, same digest) look like a model change and
/// discard every vector to recompute an identical one. So rows whose digest
/// already matches are *restamped* to the new spelling rather than
/// invalidated, which also makes the comparison converge instead of
/// re-evaluating on every boot.
///
/// When either side carries no digest — an older row stamped with a bare name,
/// or a connector that could not resolve one this boot — there is no proof of
/// sameness, so the conservative whole-string comparison still applies and the
/// row is invalidated.
pub async fn invalidate_stale_embeddings(
    pool: &PgPool,
    current_model: &str,
) -> Result<(u64, u64), String> {
    // Adopt the current spelling wherever the digest already matches. Must run
    // BEFORE the invalidation below, which would otherwise clear these rows.
    // `split_part(x, '@', 2)` yields '' when there is no '@', so the non-empty
    // test doubles as "both sides carry a digest".
    let kb_restamped = sqlx::query(
        "UPDATE knowledge_base
         SET embedding_model = $1
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model <> $1
           AND deleted_at IS NULL
           AND split_part($1, '@', 2) <> ''
           AND split_part(embedding_model, '@', 2) = split_part($1, '@', 2)",
    )
    .bind(current_model)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let tool_restamped = sqlx::query(
        "UPDATE tool_definitions
         SET embedding_model = $1
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model <> $1
           AND split_part($1, '@', 2) <> ''
           AND split_part(embedding_model, '@', 2) = split_part($1, '@', 2)",
    )
    .bind(current_model)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let restamped = kb_restamped.rows_affected() + tool_restamped.rows_affected();
    if restamped > 0 {
        tracing::info!(
            "embedding model renamed to {current_model} with an unchanged digest; \
             restamped {restamped} row(s) instead of re-embedding them"
        );
    }

    // Invalidate stale model embeddings (model mismatch).
    let kb_stale = sqlx::query(
        "UPDATE knowledge_base
         SET embedding = NULL, embedding_model = NULL
         WHERE embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND embedding_model != $1
           AND deleted_at IS NULL",
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

/// Invalidate (NULL-out) the embedding on EVERY active `knowledge_base` row,
/// regardless of model stamp or freshness, so the next backfill pass
/// re-embeds the entire knowledge base. Backs the "Recalculate Embeddings"
/// force button — for out-of-band cases (rows edited by raw SQL, corrupted
/// vectors) that the model-stamp comparison in [`invalidate_stale_embeddings`]
/// won't catch. Soft-deleted rows are skipped. Returns the row count touched.
pub async fn invalidate_all_knowledge_embeddings(pool: &PgPool) -> Result<u64, String> {
    let res = sqlx::query(
        "UPDATE knowledge_base
         SET embedding = NULL, embedding_model = NULL
         WHERE deleted_at IS NULL",
    )
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(res.rows_affected())
}

/// Backfill embeddings for `knowledge_base` rows that are missing or stale.
///
/// Each entry's content is split into chunks, all chunks are batch-embedded,
/// and the resulting vectors are stored as a `vector[]` array on the row.
///
/// Continues past batch failures so that a single bad batch does not block the
/// entire backfill.  Returns the total number of rows successfully updated.
///
/// `cancellation` is checked before each batch so an on-demand recompute (the
/// "Recalculate Embeddings" button) can be stopped via the task registry.
pub async fn backfill_knowledge_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
    cancellation: &CancellationToken,
) -> Result<usize, String> {
    let mut total = 0usize;
    let mut consecutive_failures = 0u32;

    loop {
        // Stop promptly between batches when cancelled.
        if cancellation.is_cancelled() {
            tracing::info!("knowledge embedding backfill cancelled after {total} row(s)");
            break;
        }
        // Select rows needing embedding:
        //   * never embedded / embedded by a different model
        //     (`embedding_model IS NULL OR != $1`), or
        //   * content changed since the last embed attempt
        //     (`embeddings_updated_at IS NULL OR < updated_at`) — writes bump
        //     `updated_at` but never touch the embedding, so this is how a
        //     decoupled edit gets its vector regenerated.
        //
        // Every processed row (success or failure below) gets both
        // `embedding_model` and `embeddings_updated_at = NOW()` stamped, which
        // makes all four clauses false on the next pass — so a persistently
        // failing row is attempted once per content change, not in a tight loop.
        let rows: Vec<(String, String)> = sqlx::query_as(
            // The staleness clauses are OR'd, so the soft-delete predicate has
            // to bracket them — a bare trailing AND would bind to the last OR
            // arm only and still pick up tombstones.
            "SELECT id, content FROM knowledge_base
             WHERE deleted_at IS NULL
               AND (embedding_model IS NULL
                 OR embedding_model != $1
                 OR embeddings_updated_at IS NULL
                 OR embeddings_updated_at < updated_at)
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
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((id, _), vecs) in rows.iter().zip(row_embeddings) {
                    sqlx::query(
                        "UPDATE knowledge_base
                         SET embedding = $1::vector[], embedding_model = $2,
                             embeddings_updated_at = NOW()
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
                                 SET embedding = $1::vector[], embedding_model = $2,
                                     embeddings_updated_at = NOW()
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
                            // Stamp both markers so a persistently failing row is
                            // not retried until its content changes again.
                            sqlx::query(
                                "UPDATE knowledge_base
                                 SET embedding_model = $1, embeddings_updated_at = NOW()
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
/// Each tool's text is chunked (though most will be a single chunk) and stored
/// as a `vector[]` array.
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

        // Chunk all rows and track which chunks belong to which row.
        let mut all_chunks: Vec<(usize, String)> = Vec::new();
        for (i, (_, text)) in rows.iter().enumerate() {
            for chunk in chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP) {
                all_chunks.push((i, chunk));
            }
        }

        let texts: Vec<String> = all_chunks.iter().map(|(_, t)| t.clone()).collect();
        match embed_fn(texts).await {
            Ok(embeddings) => {
                consecutive_failures = 0;
                // Group embeddings back by row index.
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((name, _), vecs) in rows.iter().zip(row_embeddings) {
                    sqlx::query(
                        "UPDATE tool_definitions
                         SET embedding = $1::vector[], embedding_model = $2
                         WHERE name = $3",
                    )
                    .bind(&vecs)
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
                    let chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    match embed_fn(chunks).await {
                        Ok(embeddings) => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            sqlx::query(
                                "UPDATE tool_definitions
                                 SET embedding = $1::vector[], embedding_model = $2
                                 WHERE name = $3",
                            )
                            .bind(&vecs)
                            .bind(current_model)
                            .bind(name)
                            .execute(pool)
                            .await
                            .map_err(|e| e.to_string())?;
                            total += 1;
                            any_succeeded = true;
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

/// Backfill NULL / stale-model embeddings for `skill_index` rows (#573),
/// mirroring [`backfill_tool_embeddings`].
///
/// The embedded text is `name + description + body` to match the row's `tsv`.
/// Each row is keyed by `(name, owner_key)` so a global and a user-scoped skill
/// sharing a name are updated independently. Returns the number of rows updated.
pub async fn backfill_skill_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;
    let mut consecutive_failures = 0u32;

    loop {
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT name, owner_key, \
                    name || ' ' || description || ' ' || coalesce(body, '') AS text \
             FROM skill_index \
             WHERE embedding_model IS NULL \
                OR embedding_model != $1 \
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
        for (i, (_, _, text)) in rows.iter().enumerate() {
            for chunk in chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP) {
                all_chunks.push((i, chunk));
            }
        }

        let texts: Vec<String> = all_chunks.iter().map(|(_, t)| t.clone()).collect();
        match embed_fn(texts).await {
            Ok(embeddings) => {
                consecutive_failures = 0;
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((name, owner_key, _), vecs) in rows.iter().zip(row_embeddings) {
                    sqlx::query(
                        "UPDATE skill_index \
                         SET embedding = $1::vector[], embedding_model = $2 \
                         WHERE name = $3 AND owner_key = $4",
                    )
                    .bind(&vecs)
                    .bind(current_model)
                    .bind(name)
                    .bind(owner_key)
                    .execute(pool)
                    .await
                    .map_err(|e| e.to_string())?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("skill embedding batch failed, retrying individually: {e}");
                let mut any_succeeded = false;
                for (name, owner_key, text) in &rows {
                    let chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    match embed_fn(chunks).await {
                        Ok(embeddings) => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            sqlx::query(
                                "UPDATE skill_index \
                                 SET embedding = $1::vector[], embedding_model = $2 \
                                 WHERE name = $3 AND owner_key = $4",
                            )
                            .bind(&vecs)
                            .bind(current_model)
                            .bind(name)
                            .bind(owner_key)
                            .execute(pool)
                            .await
                            .map_err(|e| e.to_string())?;
                            total += 1;
                            any_succeeded = true;
                        }
                        Err(e2) => {
                            tracing::warn!("skipping skill {name}: {e2}");
                            sqlx::query(
                                "UPDATE skill_index \
                                 SET embedding_model = $1 \
                                 WHERE name = $2 AND owner_key = $3",
                            )
                            .bind(current_model)
                            .bind(name)
                            .bind(owner_key)
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
                            "skill embedding backfill aborting after {consecutive_failures} consecutive failures"
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(total)
}
