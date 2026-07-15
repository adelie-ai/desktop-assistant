//! Startup embedding-backend health probe (#499).
//!
//! `EmbeddingsSettingsView.available` is a shallow connector-string check, not
//! a probe: a misconfigured embedder (for example a text-generation model that
//! answers every embed with HTTP 501) reads as healthy and silently disables
//! all vector search behind a green status. This module performs one tiny embed
//! at startup so a broken backend is caught, classified, and surfaced as a real
//! degraded-health state instead.
//!
//! The probe is model-agnostic: it catches *any* backend that cannot produce a
//! vector, regardless of the model's name. It is the primary safety net; the
//! name-based generation-model denylist in [`crate::config`] is only a faster,
//! clearer secondary guard for the common misconfiguration.

use std::time::Duration;

use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_core::ports::inbound::EmbeddingHealth;

/// The text embedded by the startup probe. Deliberately tiny — one short word
/// is enough to confirm the backend produces a vector.
const PROBE_TEXT: &str = "health";

/// Upper bound on the startup probe so a wedged backend cannot hang daemon
/// start-up indefinitely. Mirrors the per-call embed timeout used by the
/// built-in vector search.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Perform one tiny embed to verify the backend actually produces vectors, and
/// classify the outcome into an [`EmbeddingHealth`]. Success (a non-empty
/// vector) yields [`EmbeddingHealth::Ok`]; any error, timeout, or empty result
/// yields [`EmbeddingHealth::Unavailable`] carrying the reason. This never
/// returns [`EmbeddingHealth::Disabled`] — the caller sets that when no backend
/// is configured, before probing.
///
/// Model-agnostic by construction: it exercises the real embed path, so a
/// generation model that answers with HTTP 501, a wrong endpoint, or any other
/// non-embedding backend is caught here regardless of the model's name.
pub async fn probe_embedding_backend(client: &dyn EmbeddingClient) -> EmbeddingHealth {
    match tokio::time::timeout(PROBE_TIMEOUT, client.embed(vec![PROBE_TEXT.to_string()])).await {
        Ok(Ok(vectors)) if vectors.iter().any(|v| !v.is_empty()) => EmbeddingHealth::Ok,
        Ok(Ok(_)) => EmbeddingHealth::Unavailable {
            reason: "embedding backend returned no vectors".to_string(),
        },
        Ok(Err(err)) => EmbeddingHealth::Unavailable {
            reason: err.to_string(),
        },
        Err(_) => EmbeddingHealth::Unavailable {
            reason: format!("embedding probe timed out after {PROBE_TIMEOUT:?}"),
        },
    }
}

/// Assemble the health surfaced in the embeddings settings view from whether a
/// backend is configured at all (`configured`) and the startup probe result
/// (`probe`, `None` when no backend was configured):
///
/// - not configured -> [`EmbeddingHealth::Disabled`] (absent by design)
/// - configured + probe result -> that result ([`Ok`](EmbeddingHealth::Ok) or
///   [`Unavailable`](EmbeddingHealth::Unavailable))
/// - configured but never probed -> [`EmbeddingHealth::Disabled`] (defensive)
pub fn embedding_view_health(configured: bool, probe: Option<EmbeddingHealth>) -> EmbeddingHealth {
    match (configured, probe) {
        (false, _) => EmbeddingHealth::Disabled,
        (true, Some(health)) => health,
        (true, None) => EmbeddingHealth::Disabled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;

    /// Minimal [`EmbeddingClient`] that returns a preset embed outcome so the
    /// probe's classification can be exercised without a live backend.
    struct MockEmbedder {
        /// `Some(reason)` makes `embed` fail (mirroring a real HTTP-error
        /// path); `None` makes it succeed with `vectors`.
        fail_reason: Option<String>,
        vectors: Vec<Vec<f32>>,
    }

    #[async_trait::async_trait]
    impl EmbeddingClient for MockEmbedder {
        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
            match &self.fail_reason {
                Some(reason) => Err(CoreError::Llm(reason.clone())),
                None => Ok(self.vectors.clone()),
            }
        }

        async fn model_identifier(&self) -> Result<String, CoreError> {
            Ok("mock-embedder".to_string())
        }
    }

    #[tokio::test]
    async fn startup_embed_probe_marks_backend_unavailable_on_501() {
        // A generation model configured as the embedder answers every embed
        // with HTTP 501 (see `llm-ollama`'s `bail_for_status`). The probe must
        // classify that as an Unavailable health state carrying the failure
        // reason, not report the backend as healthy.
        let client = MockEmbedder {
            fail_reason: Some(
                "Ollama embeddings API error (HTTP 501 Not Implemented): not implemented"
                    .to_string(),
            ),
            vectors: Vec::new(),
        };
        let health = probe_embedding_backend(&client).await;
        match health {
            EmbeddingHealth::Unavailable { reason } => {
                assert!(
                    reason.contains("501"),
                    "reason should carry the 501 status, got: {reason}"
                );
            }
            other => panic!("expected Unavailable on 501, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn startup_embed_probe_marks_backend_ok_on_success() {
        let client = MockEmbedder {
            fail_reason: None,
            vectors: vec![vec![0.1, 0.2, 0.3]],
        };
        let health = probe_embedding_backend(&client).await;
        assert_eq!(
            health,
            EmbeddingHealth::Ok,
            "a real embedding must probe healthy"
        );
    }

    #[test]
    fn embeddings_view_reports_degraded_when_probe_fails() {
        // A configured backend whose probe failed must surface as degraded
        // (Unavailable), distinct from both healthy and absent.
        let health = embedding_view_health(
            true,
            Some(EmbeddingHealth::Unavailable {
                reason: "HTTP 501".to_string(),
            }),
        );
        match &health {
            EmbeddingHealth::Unavailable { reason } => assert!(reason.contains("501")),
            other => panic!("expected Unavailable (degraded), got {other:?}"),
        }
        assert_ne!(
            health,
            EmbeddingHealth::Disabled,
            "degraded must be distinct from disabled"
        );
        assert_ne!(health, EmbeddingHealth::Ok);
    }

    #[test]
    fn embed_backend_absent_reports_disabled_not_degraded() {
        // Anthropic has no embedding backend: the capability is absent, not
        // broken. It must report Disabled, distinct from present-but-broken
        // Unavailable.
        let health = embedding_view_health(false, None);
        assert_eq!(health, EmbeddingHealth::Disabled);
        assert!(
            !matches!(health, EmbeddingHealth::Unavailable { .. }),
            "absent backend must report Disabled, not degraded/Unavailable"
        );
    }
}
