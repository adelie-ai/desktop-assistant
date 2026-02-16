use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;

/// Outbound port for embedding text into vector representations.
pub trait EmbeddingClient: Send + Sync {
    /// Generate embeddings for a batch of texts.
    /// Returns one vector per input text.
    fn embed(
        &self,
        texts: Vec<String>,
    ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>, CoreError>> + Send;
}

/// Boxed async embedding function for passing embedding capability through
/// non-generic boundaries. Created from a concrete `EmbeddingClient` impl
/// at the daemon wiring layer.
pub type EmbedFn = Arc<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, CoreError>> + Send>>
        + Send
        + Sync,
>;
