//! Construction of the embedding client from resolved settings.
//!
//! Lives apart from `main.rs` so the mapping from a resolved
//! [`EmbeddingsSettingsView`] to a concrete client is testable.

use std::sync::Arc;

use desktop_assistant_core::ports::embedding::EmbeddingClient;

use crate::config::EmbeddingsSettingsView;

/// Build the embedding client for `view`, or `None` when embeddings are off.
pub fn build_embedding_client(view: &EmbeddingsSettingsView) -> Option<Arc<dyn EmbeddingClient>> {
    if !view.available {
        tracing::info!("embeddings unavailable (connector={})", view.connector);
        return None;
    }

    Some(match view.connector.as_str() {
        "ollama" => {
            tracing::info!("using Ollama embedding backend");
            Arc::new(desktop_assistant_llm_ollama::OllamaClient::new(
                view.base_url.clone(),
                view.model.clone(),
            ))
        }
        "bedrock" | "aws-bedrock" => {
            tracing::info!("using Bedrock embedding backend");
            Arc::new(
                desktop_assistant_llm_bedrock::BedrockClient::new(String::new())
                    .with_model(view.model.clone())
                    .with_base_url(view.base_url.clone()),
            )
        }
        "azure" => {
            // Azure serves embeddings on `/openai/v1/embeddings` (or the
            // classic deployments path). The legacy `[embeddings]` block
            // carries no surface/auth extras, so this uses the connector's
            // defaults (v1 GA, api-key auth); the deployment is the model.
            tracing::info!("using Azure embedding backend");
            Arc::new(
                desktop_assistant_llm_azure::AzureClient::new(view.api_key.clone())
                    .with_model(view.model.clone())
                    .with_base_url(view.base_url.clone()),
            )
        }
        "google" => {
            // Google embeddings target Vertex `:predict` or the Gemini API
            // `:embedContent`. The legacy `[embeddings]` block carries no
            // project/location/auth extras, so this uses the connector's
            // defaults; set the host only when explicitly configured so the
            // client can compose the Vertex host from its default location.
            tracing::info!("using Google embedding backend");
            let mut client = desktop_assistant_llm_google::GoogleClient::new(view.api_key.clone())
                .with_model(view.model.clone());
            if !view.base_url.trim().is_empty() {
                client = client.with_base_url(view.base_url.clone());
            }
            Arc::new(client)
        }
        _ => {
            tracing::info!("using OpenAI-compatible embedding backend");
            // `view.api_key` is resolved by `resolve_embeddings_config` itself
            // (purpose path uses the purpose's connection's secret/env; legacy
            // path reuses the shared LLM key when connectors match, else falls
            // back to `<CONNECTOR>_API_KEY`).
            Arc::new(
                desktop_assistant_llm_openai::OpenAiClient::new(view.api_key.clone())
                    .with_model(view.model.clone())
                    .with_base_url(view.base_url.clone()),
            )
        }
    })
}
