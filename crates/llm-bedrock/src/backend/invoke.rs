//! The Invoke API surface: `InvokeModel` on `bedrock-runtime`.
//!
//! This is the surface for everything Converse refuses. Converse is a
//! text-and-chat API, and embedding models, image and video generation models
//! and rerankers are not addressable through it at all.
//!
//! Today it serves embedding models, which is the modality this connector
//! already used `InvokeModel` for. Image and video generation are the eventual
//! driver and are not built here: `ModelKind` gains a variant when a surface
//! serves that modality, not before.

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmResponse, ModelInfo, ModelListingReport,
};
use std::sync::Arc;

use crate::backend::{BackendApiCapabilities, BedrockBackend, BedrockRequest};
use crate::sdk::SdkClients;
use crate::{BedrockEmbeddingResponse, embedding_model_from_summary};

/// The `InvokeModel` operation, as a backend.
pub(crate) struct InvokeBackend {
    sdk: Arc<SdkClients>,
}

impl InvokeBackend {
    pub(crate) fn new(sdk: Arc<SdkClients>) -> Self {
        Self { sdk }
    }

    /// Embed each text, one `InvokeModel` call per text.
    ///
    /// The body is Titan-shaped (`{"inputText": ...}`), which is what this
    /// connector has always sent. Per-model request shaping belongs here when
    /// a second embedding family arrives; until one does, a second shape would
    /// be a guess with nothing to test it against.
    pub(crate) async fn embed(
        &self,
        model: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, CoreError> {
        let client = self.sdk.runtime().await?;

        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let payload = serde_json::json!({
                "inputText": text,
            });

            let response = client
                .invoke_model()
                .model_id(model.to_string())
                .content_type("application/json")
                .accept("application/json")
                .body(payload.to_string().into_bytes().into())
                .send()
                .await
                .map_err(|e| CoreError::Llm(format!("Bedrock embeddings request failed: {e}")))?;

            let body = response.body.into_inner();
            let parsed: BedrockEmbeddingResponse = serde_json::from_slice(&body).map_err(|e| {
                CoreError::Llm(format!("failed to parse Bedrock embedding response: {e}"))
            })?;

            vectors.push(parsed.embedding);
        }

        Ok(vectors)
    }
}

#[async_trait]
impl BedrockBackend for InvokeBackend {
    fn api_name(&self) -> &'static str {
        "invoke"
    }

    fn can_serve(&self, _model_id: &str) -> bool {
        // Reach answers a **completion** request, because that is the only
        // question the selection path asks. Every model this surface serves
        // today returns vectors, and a model that returns vectors cannot serve
        // a conversation, so it reaches none of them for that purpose.
        //
        // This is what keeps "you picked an embedding model for a chat" a
        // named refusal made before the request goes out. A surface that
        // claimed reach here would take the turn and fail at AWS instead.
        //
        // It changes when this surface serves a generative modality - an image
        // model reached through `InvokeModel` - and the answer becomes the
        // model kind rather than a constant.
        false
    }

    async fn list_models(&self) -> Result<ModelListingReport, CoreError> {
        let client = self.sdk.control().await?;

        let foundation =
            client.list_foundation_models().send().await.map_err(|e| {
                CoreError::Llm(format!("Bedrock ListFoundationModels failed: {e:#}"))
            })?;

        // No inference-profile call. An embedding model is callable by its
        // bare on-demand id, so a profile for one resolves through this
        // listing or does not exist.
        let models: Vec<ModelInfo> = foundation
            .model_summaries()
            .iter()
            .filter_map(embedding_model_from_summary)
            .collect();

        Ok(ModelListingReport::complete(models))
    }

    fn capabilities(&self, _model_id: &str) -> BackendApiCapabilities {
        BackendApiCapabilities {
            // Every other field is `false` because this surface serves
            // embedding models and offers none of them for one. Claiming a
            // capability here would let selection route a turn to a surface
            // that cannot answer it.
            embeddings: true,
            streaming: false,
            tools: false,
            vision: false,
            cache_control: false,
            reasoning: false,
            hosted_tool_search: false,
        }
    }

    async fn stream_completion(
        &self,
        request: BedrockRequest,
        _on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // Unreachable through selection, which consults `can_serve` first and
        // is told this surface reaches no completion. Stated rather than
        // panicked: a future modality served here will want a real
        // implementation, and an explicit refusal is what a caller that found
        // another way in should meet.
        Err(CoreError::Llm(format!(
            "the Bedrock Invoke surface serves embedding models, so it cannot answer a \
             conversation for model {}",
            request.model
        )))
    }
}
