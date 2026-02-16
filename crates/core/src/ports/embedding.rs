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
