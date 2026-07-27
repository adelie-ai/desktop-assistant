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
            Arc::new(build_bedrock(view))
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

/// Build the Bedrock embedding client. Split out from the dispatcher above so
/// it can be inspected concretely - `Arc<dyn EmbeddingClient>` cannot be
/// downcast, and the thing worth pinning is which credential material survives
/// construction.
fn build_bedrock(view: &EmbeddingsSettingsView) -> desktop_assistant_llm_bedrock::BedrockClient {
    desktop_assistant_llm_bedrock::BedrockClient::new(String::new())
        .with_model(view.model.clone())
        .with_base_url(view.base_url.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bedrock_view() -> EmbeddingsSettingsView {
        EmbeddingsSettingsView {
            connector: "bedrock".to_string(),
            model: "amazon.titan-embed-text-v2:0".to_string(),
            base_url: "us-east-1".to_string(),
            api_key: "AKIAEXAMPLE/secret".to_string(),
            has_api_key: true,
            available: true,
            is_default: false,
        }
    }

    /// The credential is resolved all the way into the view and must reach the
    /// client. Dropping it does not fail here: the AWS SDK falls back to its
    /// default provider chain, hunts for instance metadata that a pod does not
    /// have, and reports a transport error a second later (#718).
    #[test]
    fn bedrock_embedding_client_carries_the_resolved_credential() {
        let view = bedrock_view();
        let client = build_bedrock(&view);
        assert_eq!(
            client.__api_key_for_test(),
            view.api_key,
            "the embedding client must be built with the same credential as the \
             generation client, which registry.rs passes as BedrockClient::new(resolved.api_key)"
        );
    }
}
