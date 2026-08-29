use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::CoreError;

/// Outbound port for embedding text into vector representations.
///
/// Uses [`async_trait::async_trait`] so the trait is dyn-compatible —
/// the daemon stores the active embedding backend as
/// `Option<Arc<dyn EmbeddingClient>>` (#44).
#[async_trait::async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Generate embeddings for a batch of texts.
    /// Returns one vector per input text.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError>;

    /// Return a stable identifier for the current model version.
    ///
    /// For backends where the model name is already version-pinned (OpenAI,
    /// Bedrock) this returns the model name.  For Ollama it queries the
    /// server for the model digest so that a re-pulled model is detected.
    async fn model_identifier(&self) -> Result<String, CoreError>;
}

/// Hard cap on how long an embedding call may block a real-time request. A
/// slow or wedged embedding backend (for example a stuck Ollama) must not hang
/// the turn: on timeout the caller proceeds without a vector, so a semantic
/// search falls back to full text, and a write persists unembedded for the
/// background backfill to fill in later.
pub const EMBED_TIMEOUT: Duration = Duration::from_secs(5);

/// Boxed async embedding function for passing embedding capability through
/// non-generic boundaries. Created from a concrete `EmbeddingClient` impl
/// at the daemon wiring layer.
pub type EmbedFn = Arc<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// One record's vectors and the model that produced them (#717).
///
/// One type rather than two loose fields because a search scopes its vector
/// arm to the model that produced the stored vector. A vector paired with
/// another model's name is compared against rows of another dimension, which
/// pgvector answers with an error rather than a miss -- so the two may only
/// ever be set, carried and replaced together.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkedEmbedding {
    /// One vector per content chunk, in chunk order. A short record is a
    /// single chunk; a long one is several.
    pub chunks: Vec<Vec<f32>>,
    /// Identifier of the model that produced `chunks`.
    pub model: String,
}

/// A record whose stored vector is produced from its own text.
///
/// Implemented by every write-path record that embeds inline, so
/// [`embed_chunked`] holds the all-or-nothing rule once for all of them
/// rather than once per store.
pub trait ChunkEmbeddable {
    /// The text this record is embedded from. The store's own backfill must
    /// build the same string: a vector produced from a different one is not
    /// comparable with the vectors it would be ranked against.
    fn embed_text(&self) -> String;
    /// Attach `embedding`, replacing whatever was there.
    fn set_embedding(&mut self, embedding: ChunkedEmbedding);
}

/// Embed a batch of records in place, so a record written now is semantically
/// findable now (#717).
///
/// The background backfill runs on a several-minute cadence, and the case that
/// matters is an agent looking for what it wrote moments ago -- exactly the
/// window that cadence leaves open. So the write path embeds, and the backfill
/// is the safety net rather than the only path.
///
/// Bounded by [`EMBED_TIMEOUT`]: a wedged backend must not hang the turn. On a
/// timeout, an error, or an answer that does not carry one vector per chunk,
/// every record is left unembedded and the write still lands. Those rows carry
/// a NULL vector, stay reachable through the full-text arm where their table
/// has one, and are picked up by the next backfill pass.
///
/// All-or-nothing on purpose: a short answer from the embedder would otherwise
/// be zipped chunk-to-record out of step, pairing a record with another
/// record's vector. A wrong vector is worse than no vector, because nothing
/// later detects it.
pub async fn embed_chunked<T: ChunkEmbeddable>(embed: &EmbedFn, model: &str, records: &mut [T]) {
    if records.is_empty() {
        return;
    }

    // Chunk every record, remembering which record each chunk belongs to, so
    // one backend round trip covers the whole batch.
    let mut owners: Vec<usize> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (index, record) in records.iter().enumerate() {
        for chunk in crate::chunking::chunk_text(
            &record.embed_text(),
            crate::chunking::CHUNK_MAX_CHARS,
            crate::chunking::CHUNK_OVERLAP,
        ) {
            owners.push(index);
            texts.push(chunk);
        }
    }

    let expected = texts.len();
    let vectors = match tokio::time::timeout(EMBED_TIMEOUT, embed(texts)).await {
        Ok(Ok(vectors)) if vectors.len() == expected => vectors,
        Ok(Ok(vectors)) => {
            tracing::warn!(
                returned = vectors.len(),
                expected,
                "embedder answered with the wrong number of vectors; \
                 writing the records unembedded for the backfill"
            );
            return;
        }
        Ok(Err(e)) => {
            tracing::warn!("failed to embed records: {e}");
            return;
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?EMBED_TIMEOUT,
                "embedding records timed out; writing them unembedded for the backfill"
            );
            return;
        }
    };

    let mut chunks: Vec<Vec<Vec<f32>>> = vec![Vec::new(); records.len()];
    for (index, vector) in owners.into_iter().zip(vectors) {
        chunks[index].push(vector);
    }
    for (record, chunks) in records.iter_mut().zip(chunks) {
        record.set_embedding(ChunkedEmbedding {
            chunks,
            model: model.to_string(),
        });
    }
}
