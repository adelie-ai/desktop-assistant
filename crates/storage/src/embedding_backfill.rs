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
    /// Tables the sweep could not complete, with the reason. A table listed
    /// here may still hold vectors from a superseded model, so searches over it
    /// can fail on a dimension mismatch until the next successful sweep.
    pub failed: Vec<(&'static str, String)>,
}

/// One table's contribution to a sweep.
struct TableSweep {
    restamped: u64,
    invalidated: u64,
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
/// until the row comes back still carrying it. No restore path exists today --
/// `dreaming::trash` only lists, counts and hard-reaps - so that is a guard
/// against a future one rather than a live scenario, and it costs nothing
/// meanwhile. Only rows whose
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

    // Every table is attempted even if an earlier one fails. Aborting the loop
    // on the first error would leave the remaining tables holding vectors of a
    // superseded dimension, which is exactly the breakage this sweep exists to
    // prevent -- and the tables added last are the ones that had no coverage
    // before, so a bail-out would silently restore the original bug behind a
    // single warning. Failures are named on the outcome for the caller to log.
    for table in EMBEDDED_TABLES {
        match sweep_one_table(pool, table, current_model).await {
            Ok(swept) => {
                outcome.restamped += swept.restamped;
                outcome.per_table.push((table, swept.invalidated));
            }
            Err(e) => outcome.failed.push((table, e)),
        }
    }

    for (table, error) in &outcome.failed {
        tracing::error!(
            "embedding sweep failed for {table}: {error}; rows there may still hold vectors \
             from a superseded model, and searches over them can fail on dimension mismatch"
        );
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
        // A short batch is a failed batch, not a partial success. Zipping a
        // short answer would drop the tail chunks, or -- when the shortfall
        // lands mid-batch -- pair a chunk's vector with the wrong row, and
        // nothing downstream would detect it. Route a mismatch through the
        // per-row retry, which stamps every row it touches and therefore
        // always converges.
        let batch = match embed_fn(texts).await {
            Ok(embeddings) if embeddings.len() == all_chunks.len() => Ok(embeddings),
            Ok(embeddings) => Err(format!(
                "embedder returned {} vector(s) for {} chunk(s)",
                embeddings.len(),
                all_chunks.len()
            )),
            Err(e) => Err(e),
        };

        match batch {
            Ok(embeddings) => {
                consecutive_failures = 0;
                // Group embeddings back by row index.
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((id, _), vecs) in rows.iter().zip(row_embeddings) {
                    write_knowledge_embedding(pool, id, Some(vecs), current_model).await?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("knowledge embedding batch failed, retrying individually: {e}");
                // Batch failed — retry each entry individually so good entries still get embedded.
                let mut any_succeeded = false;
                for (id, content) in &rows {
                    let chunks = chunk_text(content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    let expected = chunks.len();
                    match embed_fn(chunks).await {
                        Ok(embeddings) if embeddings.len() == expected => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            write_knowledge_embedding(pool, id, Some(vecs), current_model).await?;
                            total += 1;
                            any_succeeded = true;
                        }
                        // Both remaining arms clear the vector and stamp the
                        // model without one, so a permanently failing entry is
                        // retried once per content change rather than on every
                        // pass. They are kept apart because the operator's next
                        // step differs: a short answer points at a provider
                        // that silently caps its batch, an error at a backend
                        // that is down or rate-limiting.
                        Ok(embeddings) => {
                            tracing::warn!(
                                "skipping knowledge entry {id}: embedder returned {} vector(s) \
                                 for {expected} chunk(s)",
                                embeddings.len()
                            );
                            write_knowledge_embedding(pool, id, None, current_model).await?;
                        }
                        Err(e2) => {
                            tracing::warn!("skipping knowledge entry {id}: {e2}");
                            write_knowledge_embedding(pool, id, None, current_model).await?;
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

/// Write one knowledge-base row's vectors and model stamp.
///
/// `vectors: None` records a failed attempt: the vector is cleared and the
/// model stamped, which marks the row attempted so it is not retried in a
/// tight loop. Clearing rather than keeping a partial or stale vector matters,
/// for the reason [`write_tag_embedding`] gives -- stamping the current model
/// over a retained vector would declare it current and put it permanently
/// beyond [`invalidate_stale_embeddings`], which acts only on mismatched
/// stamps.
async fn write_knowledge_embedding(
    pool: &PgPool,
    id: &str,
    vectors: Option<Vec<Vector>>,
    current_model: &str,
) -> Result<(), String> {
    sqlx::query(
        "UPDATE knowledge_base
         SET embedding = $1::vector[], embedding_model = $2,
             embeddings_updated_at = NOW()
         WHERE id = $3",
    )
    .bind(&vectors)
    .bind(current_model)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
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
        // A short batch is a failed batch, not a partial success -- see the
        // matching comment in `backfill_knowledge_embeddings`.
        let batch = match embed_fn(texts).await {
            Ok(embeddings) if embeddings.len() == all_chunks.len() => Ok(embeddings),
            Ok(embeddings) => Err(format!(
                "embedder returned {} vector(s) for {} chunk(s)",
                embeddings.len(),
                all_chunks.len()
            )),
            Err(e) => Err(e),
        };

        match batch {
            Ok(embeddings) => {
                consecutive_failures = 0;
                // Group embeddings back by row index.
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((name, _), vecs) in rows.iter().zip(row_embeddings) {
                    write_tool_embedding(pool, name, Some(vecs), current_model).await?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("tool embedding batch failed, retrying individually: {e}");
                let mut any_succeeded = false;
                for (name, text) in &rows {
                    let chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    let expected = chunks.len();
                    match embed_fn(chunks).await {
                        Ok(embeddings) if embeddings.len() == expected => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            write_tool_embedding(pool, name, Some(vecs), current_model).await?;
                            total += 1;
                            any_succeeded = true;
                        }
                        Ok(embeddings) => {
                            tracing::warn!(
                                "skipping tool {name}: embedder returned {} vector(s) for \
                                 {expected} chunk(s)",
                                embeddings.len()
                            );
                            write_tool_embedding(pool, name, None, current_model).await?;
                        }
                        Err(e2) => {
                            tracing::warn!("skipping tool {name}: {e2}");
                            write_tool_embedding(pool, name, None, current_model).await?;
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

/// Write one tool's vectors and model stamp.
///
/// `vectors: None` records a failed attempt: the vector is cleared and the
/// model stamped, which marks the row attempted so it is not retried in a
/// tight loop. See [`write_knowledge_embedding`] for why clearing (rather than
/// keeping a partial or stale vector) matters.
async fn write_tool_embedding(
    pool: &PgPool,
    name: &str,
    vectors: Option<Vec<Vector>>,
    current_model: &str,
) -> Result<(), String> {
    sqlx::query(
        "UPDATE tool_definitions
         SET embedding = $1::vector[], embedding_model = $2
         WHERE name = $3",
    )
    .bind(&vectors)
    .bind(current_model)
    .bind(name)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
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
        // A short batch is a failed batch, not a partial success -- see the
        // matching comment in `backfill_knowledge_embeddings`.
        let batch = match embed_fn(texts).await {
            Ok(embeddings) if embeddings.len() == all_chunks.len() => Ok(embeddings),
            Ok(embeddings) => Err(format!(
                "embedder returned {} vector(s) for {} chunk(s)",
                embeddings.len(),
                all_chunks.len()
            )),
            Err(e) => Err(e),
        };

        match batch {
            Ok(embeddings) => {
                consecutive_failures = 0;
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }

                for ((name, owner_key, _), vecs) in rows.iter().zip(row_embeddings) {
                    write_skill_embedding(pool, name, owner_key, Some(vecs), current_model).await?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("skill embedding batch failed, retrying individually: {e}");
                let mut any_succeeded = false;
                for (name, owner_key, text) in &rows {
                    let chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    let expected = chunks.len();
                    match embed_fn(chunks).await {
                        Ok(embeddings) if embeddings.len() == expected => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            write_skill_embedding(pool, name, owner_key, Some(vecs), current_model)
                                .await?;
                            total += 1;
                            any_succeeded = true;
                        }
                        Ok(embeddings) => {
                            tracing::warn!(
                                "skipping skill {name}: embedder returned {} vector(s) for \
                                 {expected} chunk(s)",
                                embeddings.len()
                            );
                            write_skill_embedding(pool, name, owner_key, None, current_model)
                                .await?;
                        }
                        Err(e2) => {
                            tracing::warn!("skipping skill {name}: {e2}");
                            write_skill_embedding(pool, name, owner_key, None, current_model)
                                .await?;
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

/// Write one skill's vectors and model stamp.
///
/// `vectors: None` records a failed attempt: the vector is cleared and the
/// model stamped, which marks the row attempted so it is not retried in a
/// tight loop. See [`write_knowledge_embedding`] for why clearing (rather than
/// keeping a partial or stale vector) matters.
async fn write_skill_embedding(
    pool: &PgPool,
    name: &str,
    owner_key: &str,
    vectors: Option<Vec<Vector>>,
    current_model: &str,
) -> Result<(), String> {
    sqlx::query(
        "UPDATE skill_index \
         SET embedding = $1::vector[], embedding_model = $2 \
         WHERE name = $3 AND owner_key = $4",
    )
    .bind(&vectors)
    .bind(current_model)
    .bind(name)
    .bind(owner_key)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Backfill NULL / stale-model embeddings for `scratchpads` rows (#717),
/// mirroring [`backfill_skill_embeddings`].
///
/// The scratchpad's own write path embeds a note as it is written, because the
/// case that matters is the agent looking for what it wrote moments ago. This
/// backfill is the safety net behind that: it picks up notes the write path
/// could not embed (no backend configured, a stalled backend), and notes the
/// stale sweep cleared after a model change.
///
/// The embedded text is `note_key + content`, matching both the row's `tsv` and
/// `NewScratchpadNote::embed_text`. A vector built from a different string is
/// not comparable with the vectors it would be ranked against, so the two must
/// stay byte-identical.
///
/// Returns the number of rows updated.
pub async fn backfill_scratchpad_embeddings(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    current_model: &str,
) -> Result<usize, String> {
    let mut total = 0usize;
    let mut consecutive_failures = 0u32;

    loop {
        // Selecting on the stamp alone, never on `embedding IS NULL`, is what
        // makes this converge: the failure path below stamps the row without a
        // vector, which takes it out of this SELECT. Adding `embedding IS NULL`
        // would re-select a permanently failing row on every pass, and bill a
        // metered provider each time.
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT id, note_key || ' ' || content AS text \
             FROM scratchpads \
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
        for (i, (_, text)) in rows.iter().enumerate() {
            for chunk in chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP) {
                all_chunks.push((i, chunk));
            }
        }

        let texts: Vec<String> = all_chunks.iter().map(|(_, t)| t.clone()).collect();
        // A short batch is a failed batch, not a partial success. Zipping a
        // short answer would pair a note with another note's vector, and nothing
        // downstream detects that; route it through the per-row retry, which
        // stamps every row it touches and therefore always converges.
        let batch = match embed_fn(texts).await {
            Ok(embeddings) if embeddings.len() == all_chunks.len() => Ok(embeddings),
            Ok(embeddings) => Err(format!(
                "embedder returned {} vector(s) for {} chunk(s)",
                embeddings.len(),
                all_chunks.len()
            )),
            Err(e) => Err(e),
        };

        match batch {
            Ok(embeddings) => {
                consecutive_failures = 0;
                let mut row_embeddings: Vec<Vec<Vector>> = vec![Vec::new(); rows.len()];
                for ((row_idx, _), emb) in all_chunks.iter().zip(embeddings) {
                    row_embeddings[*row_idx].push(Vector::from(emb));
                }
                for ((id, _), vecs) in rows.iter().zip(row_embeddings) {
                    write_scratchpad_embedding(pool, id, Some(vecs), current_model).await?;
                }
                total += rows.len();
            }
            Err(e) => {
                tracing::warn!("scratchpad embedding batch failed, retrying individually: {e}");
                let mut any_succeeded = false;
                for (id, text) in &rows {
                    let chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
                    let expected = chunks.len();
                    match embed_fn(chunks).await {
                        Ok(embeddings) if embeddings.len() == expected => {
                            let vecs: Vec<Vector> =
                                embeddings.into_iter().map(Vector::from).collect();
                            write_scratchpad_embedding(pool, id, Some(vecs), current_model).await?;
                            total += 1;
                            any_succeeded = true;
                        }
                        // Both remaining arms stamp the model without a vector,
                        // so a permanently failing note is retried once per
                        // model change rather than on every pass. They are kept
                        // apart because the operator's next step differs: a
                        // short answer points at a provider that silently caps
                        // its batch, an error at a backend that is down or
                        // rate-limiting.
                        Ok(embeddings) => {
                            tracing::warn!(
                                "skipping scratchpad note {id}: embedder returned {} vector(s) \
                                 for {expected} chunk(s)",
                                embeddings.len()
                            );
                            write_scratchpad_embedding(pool, id, None, current_model).await?;
                        }
                        Err(e) => {
                            tracing::warn!("skipping scratchpad note {id}: {e}");
                            write_scratchpad_embedding(pool, id, None, current_model).await?;
                        }
                    }
                }
                if any_succeeded {
                    consecutive_failures = 0;
                } else {
                    consecutive_failures += 1;
                    if consecutive_failures >= 3 {
                        tracing::error!(
                            "scratchpad embedding backfill aborting after {consecutive_failures} consecutive failures"
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(total)
}

/// Write one note's vectors and model stamp.
///
/// `vectors: None` records a failed attempt: the vector is cleared and the model
/// stamped, which marks the row attempted so it is not retried in a tight loop.
/// Clearing rather than keeping the old vector matters, for the reason
/// [`write_tag_embedding`] gives -- stamping the current model over a retained
/// stale vector would declare it current and put it permanently beyond
/// [`invalidate_stale_embeddings`], which acts only on mismatched stamps.
async fn write_scratchpad_embedding(
    pool: &PgPool,
    id: &str,
    vectors: Option<Vec<Vector>>,
    current_model: &str,
) -> Result<(), String> {
    sqlx::query(
        "UPDATE scratchpads \
         SET embedding = $1::vector[], embedding_model = $2 \
         WHERE id = $3",
    )
    .bind(&vectors)
    .bind(current_model)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Sweep one table: restamp what only needs relabelling, invalidate what is
/// genuinely stale, and clear orphaned stamps. Table names come from
/// [`EMBEDDED_TABLES`], which holds compile-time constants and never external
/// input -- that is what makes interpolating them here safe.
async fn sweep_one_table(
    pool: &PgPool,
    table: &str,
    current_model: &str,
) -> Result<TableSweep, String> {
    // Adopt the current spelling wherever the digest already matches. Must run
    // BEFORE the invalidation below, which would otherwise clear these rows.
    // `split_part(x, '@', 2)` yields '' when there is no '@', so the non-empty
    // test doubles as "both sides carry a digest".
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

    Ok(TableSweep {
        restamped: restamped.rows_affected(),
        invalidated: stale.rows_affected() + orphan.rows_affected(),
    })
}

/// Backfill embeddings for tags whose stamp is missing or superseded.
///
/// Tags are short, so unlike the knowledge/tool/skill backfills there is no
/// chunking: one embedding per tag, stored in a scalar `vector` column rather
/// than a `vector[]`.
///
/// The embed text must stay byte-identical to the one `tag_embed_text` builds
/// for [`crate::tag_registry::create_or_match_tag`] when it looks for a
/// near-duplicate — including its rule that a tag with no description embeds as
/// its name alone. A backfilled vector is compared directly against vectors
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
            "SELECT user_id, name, \
                    CASE WHEN btrim(description) = '' \
                         THEN name ELSE name || ': ' || description END AS text \
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
        // A short batch is a failed batch, not a partial success. The port's
        // contract is one vector per input text, but no provider client
        // enforces it, and a `zip` against a short vec would leave the
        // unmatched rows unwritten -- still matching this loop's own SELECT, so
        // it would re-select them forever, spending on every pass. Route a
        // mismatch through the per-row retry below, which stamps every row it
        // touches and therefore always converges.
        let batch = match embed_fn(texts).await {
            Ok(embeddings) if embeddings.len() == rows.len() => Ok(embeddings),
            Ok(embeddings) => Err(format!(
                "embedder returned {} vector(s) for {} text(s)",
                embeddings.len(),
                rows.len()
            )),
            Err(e) => Err(e),
        };
        match batch {
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
