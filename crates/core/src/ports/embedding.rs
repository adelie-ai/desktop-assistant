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
