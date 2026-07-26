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

use crate::embedded_tables::EMBEDDED_TABLES;

/// Boxed async embedding function: takes a list of texts, returns a list of vectors.
pub type BackfillEmbedFn = Box<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, String>> + Send>>
        + Send
        + Sync,
>;

const BATCH_SIZE: i64 = 32;

/// What a stale-embedding sweep cleared, per table.
///
/// Reported per table rather than as one number so an operator can tell "the
/// model changed and 781 knowledge rows are re-embedding" from "26 tags are",
/// which have very different costs and very different recovery times.
#[derive(Debug, Default, Clone)]
pub struct StaleInvalidation {
    /// Rows invalidated in each table, in [`EMBEDDED_TABLES`] order.
    pub per_table: Vec<(&'static str, u64)>,
    /// Rows whose stamp was rewritten to the current spelling because the
    /// digest already matched, so their vectors were kept.
    pub restamped: u64,
}

impl StaleInvalidation {
    /// Total rows invalidated across every table.
    pub fn total(&self) -> u64 {
        self.per_table.iter().map(|(_, n)| n).sum()
    }

    /// Whether the sweep found nothing to do.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Per-table breakdown for a log line, listing only tables that lost rows:
    /// `knowledge_base=781, tag_registry=26`.
    pub fn summary(&self) -> String {
        self.per_table
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|(table, n)| format!("{table}={n}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

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
/// Sweeps every table in [`EMBEDDED_TABLES`], including soft-deleted knowledge
/// rows: a tombstone is invisible to search, so its stale vector looks harmless
/// until the row is restored from the trash still carrying it. Only rows whose
/// stamp is already superseded are cleared, so nothing usable is discarded, and
/// the knowledge backfill still skips tombstones (#656) -- a cleared one is not
/// re-embedded until it is actually restored.
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
) -> Result<StaleInvalidation, String> {
    let mut outcome = StaleInvalidation::default();

    for table in EMBEDDED_TABLES {
        // Adopt the current spelling wherever the digest already matches. Must
        // run BEFORE the invalidation below, which would otherwise clear these
        // rows. `split_part(x, '@', 2)` yields '' when there is no '@', so the
        // non-empty test doubles as "both sides carry a digest".
        let restamped = sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE {table}
             SET embedding_model = $1
             WHERE embedding IS NOT NULL
               AND embedding_model IS NOT NULL
               AND embedding_model <> $1
               AND split_part($1, '@', 2) <> ''
               AND split_part(embedding_model, '@', 2) = split_part($1, '@', 2)"
        )))
        .bind(current_model)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        outcome.restamped += restamped.rows_affected();

        // Invalidate stale model embeddings (model mismatch).
        let stale = sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE {table}
             SET embedding = NULL, embedding_model = NULL
             WHERE embedding IS NOT NULL
               AND embedding_model IS NOT NULL
               AND embedding_model != $1"
        )))
        .bind(current_model)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;

        // Clean up orphaned state: model is set but embedding is NULL.
        let orphan = sqlx::query(sqlx::AssertSqlSafe(format!(
            "UPDATE {table}
             SET embedding_model = NULL
             WHERE embedding IS NULL
               AND embedding_model IS NOT NULL"
        )))
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;

        outcome
            .per_table
            .push((table, stale.rows_affected() + orphan.rows_affected()));
    }

    if outcome.restamped > 0 {
        tracing::info!(
            "embedding model renamed to {current_model} with an unchanged digest; \
             restamped {} row(s) instead of re-embedding them",
            outcome.restamped
        );
    }

    Ok(outcome)
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

/// Backfill embeddings for tags whose stamp is missing or superseded.
///
/// Tags are short, so unlike the knowledge/tool/skill backfills there is no
/// chunking: one embedding per tag, stored in a scalar `vector` column rather
/// than a `vector[]`.
///
/// The embed text must stay byte-identical to the one
/// [`crate::tag_registry::create_or_match_tag`] builds when it looks for a
/// near-duplicate. A backfilled vector is compared directly against vectors
/// produced by that path, so embedding a different string here would make the
/// dedup distances meaningless rather than merely imprecise.
///
/// Deprecated tags are skipped: they are excluded from the dedup search, so
/// re-embedding them is spend with no reader.
pub async fn backfill_tag_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;
    let mut consecutive_failures = 0u32;

    loop {
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT user_id, name, name || ': ' || description AS text \
             FROM tag_registry \
             WHERE deprecated_for_tag IS NULL \
               AND (embedding IS NULL \
                 OR embedding_model IS NULL \
                 OR embedding_model != $1) \
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

        let texts: Vec<String> = rows.iter().map(|(_, _, text)| text.clone()).collect();
        match embed_fn(texts).await {
            Ok(embeddings) => {
                consecutive_failures = 0;
                for ((user_id, name, _), embedding) in rows.iter().zip(embeddings) {
                    write_tag_embedding(pool, user_id, name, Some(embedding), current_model)
                        .await?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("tag embedding batch failed, retrying individually: {e}");
                let mut any_succeeded = false;
                for (user_id, name, text) in &rows {
                    match embed_fn(vec![text.clone()]).await {
                        Ok(embeddings) => {
                            let embedding = embeddings.into_iter().next();
                            write_tag_embedding(pool, user_id, name, embedding, current_model)
                                .await?;
                            total += 1;
                            any_succeeded = true;
                        }
                        Err(e2) => {
                            // Stamp the model without a vector so a permanently
                            // failing tag is retried once per model change
                            // rather than on every pass.
                            tracing::warn!("skipping tag {name}: {e2}");
                            write_tag_embedding(pool, user_id, name, None, current_model).await?;
                        }
                    }
                }
                if any_succeeded {
                    consecutive_failures = 0;
                } else {
                    consecutive_failures += 1;
                    if consecutive_failures >= 3 {
                        tracing::error!(
                            "tag embedding backfill aborting after {consecutive_failures} consecutive failures"
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(total)
}

/// Write one tag's vector and model stamp.
///
/// `embedding: None` records a failed attempt: the vector is cleared and the
/// model stamped, which marks the row attempted so it is not retried in a tight
/// loop. Clearing rather than keeping the old vector matters -- a row only
/// reaches the backfill when its embedding is absent or its stamp superseded,
/// so whatever is there cannot be compared against a current-model query
/// anyway. Stamping the current model over a retained stale vector would
/// declare it current and put it permanently beyond
/// [`invalidate_stale_embeddings`], which only looks at mismatched stamps.
async fn write_tag_embedding(
    pool: &PgPool,
    user_id: &str,
    name: &str,
    embedding: Option<Vec<f32>>,
    current_model: &str,
) -> Result<(), String> {
    let vector = embedding.map(Vector::from);
    sqlx::query(
        "UPDATE tag_registry \
         SET embedding = $1, embedding_model = $2 \
         WHERE user_id = $3 AND name = $4",
    )
    .bind(&vector)
    .bind(current_model)
    .bind(user_id)
    .bind(name)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}
