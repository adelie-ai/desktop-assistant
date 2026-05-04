use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

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

/// Boxed async embedding function for passing embedding capability through
/// non-generic boundaries. Created from a concrete `EmbeddingClient` impl
/// at the daemon wiring layer.
pub type EmbedFn = Arc<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, CoreError>> + Send>>
        + Send
        + Sync,
>;
