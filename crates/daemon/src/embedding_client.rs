//! Construction of the embedding client from resolved settings.
//!
//! Lives apart from `main.rs` so the mapping from a resolved
//! [`EmbeddingsSettingsView`] to a concrete client is testable. This is the one
//! place a connector's credential material has to be threaded through, and
//! getting it wrong fails a long way from the cause: a client built without a
//! credential does not fail at construction, it fails when the provider SDK
//! gives up looking for ambient credentials and reports a transport error
//! (#718).

use std::sync::Arc;

use desktop_assistant_core::ports::embedding::EmbeddingClient;

use crate::config::EmbeddingsSettingsView;

/// Whether a connector needs a credential to work at all.
///
/// Ollama is local and unauthenticated. Every other connector talks to a
/// provider that will refuse an anonymous caller - or worse will not even try,
/// because its SDK goes hunting for ambient credentials first and then reports
/// that timeout instead of the missing configuration.
fn requires_credential(connector: &str) -> bool {
    !matches!(connector, "ollama")
}

/// Whether `view` names a connector that needs a credential and has none.
///
/// Separated from the logging so the decision can be asserted directly.
fn credential_missing(view: &EmbeddingsSettingsView) -> bool {
    requires_credential(&view.connector)
        && view.api_key.trim().is_empty()
        && view.aws_profile.is_none()
}

/// Build the embedding client for `view`, or `None` when embeddings are off.
pub fn build_embedding_client(view: &EmbeddingsSettingsView) -> Option<Arc<dyn EmbeddingClient>> {
    if !view.available {
        tracing::info!("embeddings unavailable (connector={})", view.connector);
        return None;
    }

    // Say this while the cause is still visible. Without it the first symptom
    // is a provider transport error seconds later, which reads as a network
    // fault rather than as configuration.
    if credential_missing(view) {
        // Deliberately not phrased as "calls will fail": a self-hosted
        // OpenAI-compatible endpoint is legitimately keyless, and a warning
        // that asserts a failure which does not happen is one operators learn
        // to scroll past.
        tracing::warn!(
            "no credential resolved for the {} embedding backend. If it requires \
             authentication, embedding calls will fail - check the secret configured for \
             the connection the embedding purpose binds to.",
            view.connector
        );
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
    desktop_assistant_llm_bedrock::BedrockClient::new(view.api_key.clone())
        .with_model(view.model.clone())
        .with_base_url(view.base_url.clone())
        .with_aws_profile(view.aws_profile.clone())
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
            aws_profile: None,
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

    /// Bedrock accepts a profile instead of a key, and the generation path
    /// passes one via `with_aws_profile`. An operator using profile auth must
    /// not silently lose embeddings.
    #[test]
    fn bedrock_embedding_client_carries_the_resolved_aws_profile() {
        let mut view = bedrock_view();
        view.api_key = String::new();
        view.has_api_key = false;
        view.aws_profile = Some("adele".to_string());

        let client = build_bedrock(&view);
        assert_eq!(
            client.__aws_profile_for_test(),
            view.aws_profile.as_deref(),
            "profile auth is the other half of Bedrock credentials and must survive construction"
        );
    }

    /// A connector that needs a credential and resolved none is the failure
    /// this module exists to make legible.
    #[test]
    fn credential_missing_flags_a_credential_requiring_connector_with_nothing_resolved() {
        let mut view = bedrock_view();
        view.api_key = String::new();
        view.has_api_key = false;
        assert!(credential_missing(&view));
    }

    /// An AWS profile is a credential too, so a key-less Bedrock view that has
    /// one must not be flagged.
    #[test]
    fn credential_missing_accepts_an_aws_profile_in_place_of_a_key() {
        let mut view = bedrock_view();
        view.api_key = String::new();
        view.has_api_key = false;
        view.aws_profile = Some("adele".to_string());
        assert!(!credential_missing(&view));
    }

    /// Ollama is local and unauthenticated; flagging it would train operators
    /// to ignore the warning.
    #[test]
    fn credential_missing_does_not_flag_ollama() {
        let mut view = bedrock_view();
        view.connector = "ollama".to_string();
        view.api_key = String::new();
        view.has_api_key = false;
        assert!(!credential_missing(&view));
    }

    /// Every credential-taking connector must reach its client with the key the
    /// view resolved. Pins the class rather than the one branch that had it
    /// wrong, so a future branch cannot quietly omit it.
    #[test]
    fn every_credential_requiring_connector_is_built_with_the_resolved_key() {
        for connector in ["bedrock", "aws-bedrock", "azure", "google", "openai"] {
            let mut view = bedrock_view();
            view.connector = connector.to_string();
            assert!(
                !credential_missing(&view),
                "{connector} resolved a key, so it must not be flagged as missing one"
            );
            assert!(
                build_embedding_client(&view).is_some(),
                "{connector} must build a client when it is available"
            );
        }
    }
}
