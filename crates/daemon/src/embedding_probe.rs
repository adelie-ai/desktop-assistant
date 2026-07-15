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
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Minimal [`EmbeddingClient`] that returns a preset embed outcome so the
    /// probe's classification can be exercised without a live backend.
    ///
    /// It can also simulate a *cold* backend: the first `slow_calls` calls sleep
    /// for `cold_delay` (mimicking a model loading into memory on the first
    /// embed) before returning. That lets the retry/timeout path be tested
    /// deterministically with tiny durations, so no test ever waits real
    /// seconds.
    #[derive(Default)]
    struct MockEmbedder {
        /// `Some(reason)` makes `embed` fail (mirroring a real HTTP-error
        /// path); `None` makes it succeed with `vectors`.
        fail_reason: Option<String>,
        vectors: Vec<Vec<f32>>,
        /// Sleep this long on each of the first `slow_calls` calls.
        cold_delay: Duration,
        /// Number of leading calls that "cold-load" (sleep `cold_delay`).
        slow_calls: usize,
        /// Total calls observed, so a test can assert a retry actually re-called.
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl EmbeddingClient for MockEmbedder {
        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.slow_calls {
                tokio::time::sleep(self.cold_delay).await;
            }
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
            ..Default::default()
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
            ..Default::default()
        };
        let health = probe_embedding_backend(&client).await;
        assert_eq!(
            health,
            EmbeddingHealth::Ok,
            "a real embedding must probe healthy"
        );
    }

    #[tokio::test]
    async fn startup_embed_probe_marks_backend_unavailable_on_empty_vectors() {
        // #499's core failure mode: the backend answers HTTP 200 but produces no
        // usable embedding. Both an empty outer vec and a vec-of-empty-vec must
        // classify Unavailable (not a false-green Ok) — there is no vector to
        // search with either way.
        for vectors in [Vec::new(), vec![Vec::new()]] {
            let client = MockEmbedder {
                fail_reason: None,
                vectors,
                ..Default::default()
            };
            let health = probe_embedding_backend(&client).await;
            match health {
                EmbeddingHealth::Unavailable { reason } => assert!(
                    reason.contains("no vectors"),
                    "empty embedding must classify Unavailable, got reason: {reason}"
                ),
                other => panic!("expected Unavailable on empty vectors, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn startup_embed_probe_times_out_to_unavailable() {
        // A backend that never answers within the per-attempt timeout must
        // classify Unavailable carrying a timeout reason — not hang startup, not
        // read as healthy. Uses a TINY injected timeout and a single attempt so
        // the test is fast: no real multi-second sleeps.
        let client = MockEmbedder {
            fail_reason: None,
            vectors: vec![vec![0.1_f32]],
            cold_delay: Duration::from_secs(3600),
            slow_calls: usize::MAX,
            ..Default::default()
        };
        let health = probe_embedding_backend_with(&client, Duration::from_millis(10), 1).await;
        match health {
            EmbeddingHealth::Unavailable { reason } => assert!(
                reason.contains("timed out"),
                "timeout must classify Unavailable with a timeout reason, got: {reason}"
            ),
            other => panic!("expected Unavailable on timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn startup_embed_probe_tolerates_cold_first_embed() {
        // Regression guard for the cold-start bug: on a slow/cold backend (e.g.
        // Ollama loading the embed model into memory on the NUC) the FIRST embed
        // can exceed the per-attempt timeout. The probe must RETRY rather than
        // permanently disabling a healthy-but-cold backend. Here the first call
        // "cold loads" past a tiny timeout; the second returns a real vector -> Ok.
        let client = MockEmbedder {
            fail_reason: None,
            vectors: vec![vec![0.1, 0.2, 0.3]],
            cold_delay: Duration::from_millis(200),
            slow_calls: 1,
            ..Default::default()
        };
        let health = probe_embedding_backend_with(&client, Duration::from_millis(20), 3).await;
        assert_eq!(
            health,
            EmbeddingHealth::Ok,
            "a healthy-but-cold backend must survive the first slow embed via retry"
        );
        assert!(
            client.calls.load(Ordering::SeqCst) >= 2,
            "the probe must have retried after the cold first embed"
        );
    }

    #[tokio::test]
    async fn startup_embed_probe_does_not_retry_hard_error() {
        // A definitively-down backend (immediate error — HTTP 501, connection
        // refused) must be classified Unavailable on the FIRST attempt without
        // burning the retry budget on repeated calls, so a genuinely broken
        // backend fails fast.
        let client = MockEmbedder {
            fail_reason: Some("HTTP 501 Not Implemented".to_string()),
            vectors: Vec::new(),
            ..Default::default()
        };
        let health = probe_embedding_backend_with(&client, Duration::from_secs(30), 5).await;
        assert!(matches!(health, EmbeddingHealth::Unavailable { .. }));
        assert_eq!(
            client.calls.load(Ordering::SeqCst),
            1,
            "a hard error must not be retried"
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
    fn embedding_view_health_configured_but_unprobed_is_unknown() {
        // A backend is configured but no probe result is available (probing was
        // skipped, or the probe handle was not wired). The honest state is
        // Unknown — health was not determined — NOT Disabled, which would
        // misreport a configured backend as off by design.
        let health = embedding_view_health(true, None);
        assert_eq!(health, EmbeddingHealth::Unknown);
        assert_ne!(health, EmbeddingHealth::Disabled);
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

    #[test]
    fn keep_embedding_client_only_on_ok() {
        // Pin the honest-FTS wiring: only a healthy probe keeps the embedding
        // client. Every other state drops it so downstream vector paths take the
        // disabled -> full-text-search route uniformly.
        assert!(keep_embedding_client(&EmbeddingHealth::Ok));
        assert!(!keep_embedding_client(&EmbeddingHealth::Disabled));
        assert!(!keep_embedding_client(&EmbeddingHealth::Unknown));
        assert!(!keep_embedding_client(&EmbeddingHealth::Unavailable {
            reason: "HTTP 501".to_string(),
        }));
    }
}
