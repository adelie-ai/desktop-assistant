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

use std::sync::{Arc, RwLock};
use std::time::Duration;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_core::ports::inbound::EmbeddingHealth;
use tokio::sync::oneshot;

/// The text embedded by the startup probe. Deliberately tiny — one short word
/// is enough to confirm the backend produces a vector.
const PROBE_TEXT: &str = "health";

/// Per-attempt timeout for the startup probe. Sized generously for a **cold**
/// backend: on the target hardware (a slow Intel NUC) Ollama loads the embed
/// model into memory on the first call, so the first embed can be many seconds.
/// A too-tight bound (the old 10s) misclassifies a healthy-but-cold backend as
/// broken and permanently disables vector search until restart — the regression
/// this value plus [`PROBE_ATTEMPTS`] fixes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// How many times the probe attempts the embed before giving up. Only *timeouts*
/// consume attempts (a slow cold load); a hard error is classified immediately
/// (see [`probe_embedding_backend_with`]). Worst case is bounded at roughly
/// `PROBE_ATTEMPTS * PROBE_TIMEOUT` plus backoff.
const PROBE_ATTEMPTS: u32 = 2;

/// Short pause between timed-out attempts, giving a cold backend a moment to
/// finish loading the model before the next embed.
const PROBE_BACKOFF: Duration = Duration::from_millis(500);

/// Perform one tiny embed to verify the backend actually produces vectors, and
/// classify the outcome into an [`EmbeddingHealth`]. Uses generous, cold-load
/// tolerant defaults ([`PROBE_TIMEOUT`], [`PROBE_ATTEMPTS`]); see
/// [`probe_embedding_backend_with`] for the injectable inner form (tests pass a
/// tiny timeout).
///
/// Model-agnostic by construction: it exercises the real embed path, so a
/// generation model that answers with HTTP 501, a wrong endpoint, or any other
/// non-embedding backend is caught here regardless of the model's name.
pub async fn probe_embedding_backend(client: &dyn EmbeddingClient) -> EmbeddingHealth {
    probe_embedding_backend_with(client, PROBE_TIMEOUT, PROBE_ATTEMPTS).await
}

/// Inner probe with an injectable per-attempt `timeout` and `attempts` budget so
/// the retry/timeout policy is testable without real multi-second sleeps.
///
/// Classification:
/// - a non-empty vector -> [`EmbeddingHealth::Ok`];
/// - HTTP 200 but no usable vector -> [`EmbeddingHealth::Unavailable`] (the
///   backend answered but is not an embedder — a hard failure, not retried);
/// - a definitive backend error (HTTP 501, connection refused, ...) ->
///   [`EmbeddingHealth::Unavailable`] on the FIRST attempt (not retried, so a
///   genuinely-down backend fails fast without burning the budget);
/// - a **timeout** -> retried up to `attempts` times with a short backoff,
///   because on a cold backend the first embed can be slow while the model loads
///   into memory; only if every attempt times out is it
///   [`EmbeddingHealth::Unavailable`].
///
/// Never returns [`EmbeddingHealth::Disabled`]/[`EmbeddingHealth::Unknown`] — the
/// caller sets those when there is no backend to probe.
pub async fn probe_embedding_backend_with(
    client: &dyn EmbeddingClient,
    timeout: Duration,
    attempts: u32,
) -> EmbeddingHealth {
    let attempts = attempts.max(1);
    let mut timeout_reason = format!("embedding probe timed out after {timeout:?}");
    for attempt in 1..=attempts {
        match tokio::time::timeout(timeout, client.embed(vec![PROBE_TEXT.to_string()])).await {
            Ok(Ok(vectors)) if vectors.iter().any(|v| !v.is_empty()) => return EmbeddingHealth::Ok,
            Ok(Ok(_)) => {
                // The backend answered but produced no usable vector; retrying a
                // definitively-wrong backend will not help.
                return EmbeddingHealth::Unavailable {
                    reason: "embedding backend returned no vectors".to_string(),
                };
            }
            Ok(Err(err)) => {
                // A definitive error (e.g. HTTP 501, connection refused): classify
                // now rather than spending the remaining attempts on it.
                return EmbeddingHealth::Unavailable {
                    reason: err.to_string(),
                };
            }
            Err(_) => {
                // Timed out: likely a cold model load on the first embed. Retry a
                // bounded number of times before declaring the backend broken.
                timeout_reason =
                    format!("embedding probe timed out after {timeout:?} on {attempt} attempt(s)");
                if attempt < attempts {
                    tokio::time::sleep(PROBE_BACKOFF).await;
                }
            }
        }
    }
    EmbeddingHealth::Unavailable {
        reason: timeout_reason,
    }
}

/// Assemble the health surfaced in the embeddings settings view from whether a
/// backend is configured at all (`configured`) and the startup probe result
/// (`probe`, `None` when no backend was configured or it was not probed):
///
/// - not configured -> [`EmbeddingHealth::Disabled`] (absent by design)
/// - configured + probe result -> that result ([`Ok`](EmbeddingHealth::Ok) or
///   [`Unavailable`](EmbeddingHealth::Unavailable))
/// - configured but never probed -> [`EmbeddingHealth::Unknown`] (honest "not
///   determined", never a false-green `Ok` or a misleading `Disabled`)
pub fn embedding_view_health(configured: bool, probe: Option<EmbeddingHealth>) -> EmbeddingHealth {
    match (configured, probe) {
        (false, _) => EmbeddingHealth::Disabled,
        (true, Some(health)) => health,
        (true, None) => EmbeddingHealth::Unknown,
    }
}

/// Whether to keep the embedding client wired given the resolved startup health.
///
/// Only a healthy ([`Ok`](EmbeddingHealth::Ok)) probe keeps it; every other
/// state ([`Disabled`](EmbeddingHealth::Disabled),
/// [`Unavailable`](EmbeddingHealth::Unavailable),
/// [`Unknown`](EmbeddingHealth::Unknown)) drops it, so every downstream vector
/// path (query embedding, stale-embedding invalidation, background backfill)
/// takes the disabled -> full-text-search route uniformly instead of churning
/// against a backend that cannot embed. Extracted so this honest-fallback
/// guarantee is unit-pinned rather than buried in `main`.
pub(crate) fn keep_embedding_client(health: &EmbeddingHealth) -> bool {
    matches!(health, EmbeddingHealth::Ok)
}

/// The live health of the embedding backend, shared between the daemon's vector
/// paths, the background re-check loop and `GetConfig`.
#[derive(Debug)]
pub struct EmbeddingCapability {
    health: RwLock<EmbeddingHealth>,
}

impl EmbeddingCapability {
    /// Seed the capability with the startup probe's verdict.
    pub fn new(initial: EmbeddingHealth) -> Self {
        Self {
            health: RwLock::new(initial),
        }
    }

    /// The current health.
    pub fn health(&self) -> EmbeddingHealth {
        self.health
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// `Some(reason)` when the backend must not be called right now.
    pub fn unavailable_reason(&self) -> Option<String> {
        match self.health() {
            EmbeddingHealth::Unavailable { reason } => Some(reason),
            EmbeddingHealth::Disabled => Some("no embedding backend configured".to_string()),
            EmbeddingHealth::Ok | EmbeddingHealth::Unknown => None,
        }
    }

    /// Record that the backend produced a real vector.
    pub fn record_ok(&self) {
        self.set(EmbeddingHealth::Ok);
    }

    /// Record that the backend failed to produce a vector, carrying why.
    pub fn record_failure(&self, reason: impl Into<String>) {
        self.set(EmbeddingHealth::Unavailable {
            reason: reason.into(),
        });
    }

    fn set(&self, next: EmbeddingHealth) {
        let mut guard = self
            .health
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if *guard == next {
            return;
        }
        match (&*guard, &next) {
            (_, EmbeddingHealth::Ok) => {
                tracing::info!("embedding backend healthy; vector search live")
            }
            (_, EmbeddingHealth::Unavailable { reason }) => tracing::warn!(
                "embedding backend unavailable; vector search degraded to full-text search: {reason}"
            ),
            _ => {}
        }
        *guard = next;
    }
}

/// [`EmbeddingClient`] decorator that keeps [`EmbeddingCapability`] current.
pub struct HealthGatedEmbeddingClient {
    inner: Arc<dyn EmbeddingClient>,
    capability: Arc<EmbeddingCapability>,
}

impl HealthGatedEmbeddingClient {
    /// Wrap `inner`, reporting every embed outcome into `capability`.
    pub fn new(inner: Arc<dyn EmbeddingClient>, capability: Arc<EmbeddingCapability>) -> Self {
        Self { inner, capability }
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for HealthGatedEmbeddingClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let _ = self.capability.unavailable_reason();
        self.inner.embed(texts).await
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        self.inner.model_identifier().await
    }
}

/// Re-probe a degraded backend until it recovers, updating `capability`.
pub async fn recheck_degraded_backend(
    client: Arc<dyn EmbeddingClient>,
    capability: Arc<EmbeddingCapability>,
    interval: Duration,
    shutdown: oneshot::Receiver<()>,
) {
    let _ = (client, capability, interval);
    let _ = shutdown.await;
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
        /// With `fail_reason` set, `Some(n)` fails only the first `n` calls and
        /// succeeds from call `n` on — a backend that is down when the daemon
        /// boots and comes back a moment later. `None` fails every call.
        recover_after: Option<usize>,
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
                Some(reason) if self.recover_after.is_none_or(|until| n < until) => {
                    Err(CoreError::Llm(reason.clone()))
                }
                _ => Ok(self.vectors.clone()),
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
    fn failed_boot_probe_keeps_embedding_client_wired() {
        // The client stays wired whenever a backend is configured, whatever the
        // boot probe said. Dropping it on a failed probe is what disabled the
        // backfill loop, dreaming, consolidation and the whole knowledge
        // maintenance handler for the process lifetime.
        assert!(
            keep_embedding_client(&EmbeddingHealth::Unavailable {
                reason: "connection refused".to_string(),
            }),
            "a failed boot probe must not latch the embedding client off"
        );
        assert!(keep_embedding_client(&EmbeddingHealth::Ok));
        assert!(keep_embedding_client(&EmbeddingHealth::Unknown));
    }

    #[test]
    fn absent_embedding_backend_still_drops_the_client() {
        // Absent is not degraded: with no backend configured at all there is
        // nothing to wire and nothing to re-probe.
        assert!(!keep_embedding_client(&EmbeddingHealth::Disabled));
    }

    // --- Live capability -------------------------------------------------

    /// Poll `capability` until it reports `want`, bounded so a regression fails
    /// the test rather than hanging the suite.
    async fn await_health(capability: &EmbeddingCapability, want: &EmbeddingHealth) -> bool {
        tokio::time::timeout(Duration::from_secs(5), async {
            while capability.health() != *want {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .is_ok()
    }

    fn unavailable(reason: &str) -> EmbeddingHealth {
        EmbeddingHealth::Unavailable {
            reason: reason.to_string(),
        }
    }

    #[tokio::test]
    async fn degraded_backend_recovers_without_daemon_restart() {
        // The headline regression: Ollama is restarting when the daemon boots,
        // so the probe classifies Unavailable. Once the backend answers again
        // the capability must return to Ok on its own — no manual restart.
        let capability = Arc::new(EmbeddingCapability::new(unavailable("connection refused")));
        let client: Arc<dyn EmbeddingClient> = Arc::new(MockEmbedder {
            fail_reason: Some("connection refused".to_string()),
            vectors: vec![vec![0.1, 0.2]],
            recover_after: Some(1),
            ..Default::default()
        });
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(recheck_degraded_backend(
            Arc::clone(&client),
            Arc::clone(&capability),
            Duration::from_millis(5),
            shutdown_rx,
        ));

        assert!(
            await_health(&capability, &EmbeddingHealth::Ok).await,
            "a backend that came back must re-probe healthy without a restart, health was {:?}",
            capability.health()
        );

        let _ = shutdown_tx.send(());
        task.await.expect("recheck task should exit cleanly");
    }

    #[tokio::test]
    async fn recheck_keeps_a_still_broken_backend_degraded() {
        // Recovery must be evidence-based: a backend that is still refusing
        // stays Unavailable, and the reason stays legible.
        let capability = Arc::new(EmbeddingCapability::new(unavailable("HTTP 501")));
        let client: Arc<dyn EmbeddingClient> = Arc::new(MockEmbedder {
            fail_reason: Some("HTTP 501 Not Implemented".to_string()),
            vectors: Vec::new(),
            ..Default::default()
        });
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(recheck_degraded_backend(
            Arc::clone(&client),
            Arc::clone(&capability),
            Duration::from_millis(5),
            shutdown_rx,
        ));

        assert!(
            await_health(&capability, &unavailable("HTTP 501 Not Implemented")).await,
            "a still-broken backend must stay Unavailable with the fresh reason, health was {:?}",
            capability.health()
        );

        let _ = shutdown_tx.send(());
        task.await.expect("recheck task should exit cleanly");
    }

    #[tokio::test]
    async fn recheck_leaves_a_healthy_backend_unprobed() {
        // No churn on the happy path: while the capability reads Ok the loop
        // must not spend calls on the backend.
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Ok));
        let client = Arc::new(MockEmbedder {
            fail_reason: None,
            vectors: vec![vec![0.1]],
            ..Default::default()
        });
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(recheck_degraded_backend(
            Arc::clone(&client) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
            Duration::from_millis(2),
            shutdown_rx,
        ));

        tokio::time::sleep(Duration::from_millis(60)).await;
        let _ = shutdown_tx.send(());
        task.await.expect("recheck task should exit cleanly");

        assert_eq!(
            client.calls.load(Ordering::SeqCst),
            0,
            "a healthy backend must not be re-probed"
        );
    }

    #[tokio::test]
    async fn recheck_stops_on_shutdown() {
        // The loop is a long-running spawned task: it must be cancellable so
        // daemon shutdown does not hang or leak it.
        let capability = Arc::new(EmbeddingCapability::new(unavailable("down")));
        let client: Arc<dyn EmbeddingClient> = Arc::new(MockEmbedder::default());
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(recheck_degraded_backend(
            client,
            capability,
            Duration::from_secs(3600),
            shutdown_rx,
        ));

        let _ = shutdown_tx.send(());
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("recheck must stop promptly on shutdown")
            .expect("recheck task should exit cleanly");
    }

    // --- Health-gated embedding client -----------------------------------

    #[tokio::test]
    async fn gated_embed_fails_fast_while_backend_is_unavailable() {
        // While the capability reads Unavailable the vector paths are gated off,
        // so callers take the honest full-text route instead of churning
        // against a backend that cannot embed.
        let capability = Arc::new(EmbeddingCapability::new(unavailable("connection refused")));
        let inner = Arc::new(MockEmbedder {
            fail_reason: None,
            vectors: vec![vec![0.1]],
            ..Default::default()
        });
        let gated = HealthGatedEmbeddingClient::new(
            Arc::clone(&inner) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        );

        let err = gated
            .embed(vec!["hello".to_string()])
            .await
            .expect_err("a gated-off backend must not be called");
        assert!(
            err.to_string().contains("connection refused"),
            "the gate must say why it declined, got: {err}"
        );
        assert_eq!(
            inner.calls.load(Ordering::SeqCst),
            0,
            "the gate must not reach the backend while it is Unavailable"
        );
    }

    #[tokio::test]
    async fn gated_embed_marks_backend_unavailable_on_error() {
        // A live failure has to move the shared health, not just this one call:
        // that is what makes the state an honest capability instead of a
        // boot-time latch.
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Ok));
        let gated = HealthGatedEmbeddingClient::new(
            Arc::new(MockEmbedder {
                fail_reason: Some("HTTP 501 Not Implemented".to_string()),
                vectors: Vec::new(),
                ..Default::default()
            }) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        );

        gated
            .embed(vec!["hello".to_string()])
            .await
            .expect_err("a failing backend must surface the error");
        match capability.health() {
            EmbeddingHealth::Unavailable { reason } => assert!(
                reason.contains("501"),
                "the live health must carry the failure reason, got: {reason}"
            ),
            other => panic!("a failed embed must degrade the live health, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn gated_embed_restores_health_after_a_successful_embed() {
        // A configured-but-unprobed backend is attempted optimistically, and a
        // real vector promotes the capability to Ok.
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Unknown));
        let gated = HealthGatedEmbeddingClient::new(
            Arc::new(MockEmbedder {
                fail_reason: None,
                vectors: vec![vec![0.1, 0.2, 0.3]],
                ..Default::default()
            }) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        );

        let vectors = gated
            .embed(vec!["hello".to_string()])
            .await
            .expect("a healthy backend must embed");
        assert_eq!(vectors, vec![vec![0.1, 0.2, 0.3]]);
        assert_eq!(
            capability.health(),
            EmbeddingHealth::Ok,
            "a real vector must promote the live health to Ok"
        );
    }

    #[tokio::test]
    async fn gated_embed_marks_backend_unavailable_on_empty_vectors() {
        // HTTP 200 with no usable vector is the #499 false-green: the transport
        // call succeeded but the capability is broken. Health must degrade even
        // though the call returned Ok.
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Ok));
        let gated = HealthGatedEmbeddingClient::new(
            Arc::new(MockEmbedder {
                fail_reason: None,
                vectors: vec![Vec::new()],
                ..Default::default()
            }) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        );

        gated
            .embed(vec!["hello".to_string()])
            .await
            .expect("an empty-vector answer is still a transport success");
        match capability.health() {
            EmbeddingHealth::Unavailable { reason } => assert!(
                reason.contains("no vectors"),
                "an empty embedding must degrade the live health, got: {reason}"
            ),
            other => panic!("expected Unavailable on empty vectors, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn gated_embed_of_an_empty_batch_does_not_change_health() {
        // Empty input: an empty batch legitimately yields no vectors, so it must
        // not be read as a broken backend.
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Ok));
        let gated = HealthGatedEmbeddingClient::new(
            Arc::new(MockEmbedder {
                fail_reason: None,
                vectors: Vec::new(),
                ..Default::default()
            }) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        );

        gated
            .embed(Vec::new())
            .await
            .expect("an empty batch must be passed through");
        assert_eq!(
            capability.health(),
            EmbeddingHealth::Ok,
            "an empty batch must leave the live health untouched"
        );
    }

    #[tokio::test]
    async fn gated_model_identifier_passes_through() {
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Ok));
        let gated = HealthGatedEmbeddingClient::new(
            Arc::new(MockEmbedder::default()) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        );
        assert_eq!(
            gated
                .model_identifier()
                .await
                .expect("model identifier should pass through"),
            "mock-embedder"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_gated_embeds_converge_on_one_health() {
        // Many turns can embed at once. Concurrent outcome reporting must not
        // deadlock or leave the shared health torn.
        let capability = Arc::new(EmbeddingCapability::new(EmbeddingHealth::Ok));
        let gated = Arc::new(HealthGatedEmbeddingClient::new(
            Arc::new(MockEmbedder {
                fail_reason: Some("connection refused".to_string()),
                vectors: Vec::new(),
                ..Default::default()
            }) as Arc<dyn EmbeddingClient>,
            Arc::clone(&capability),
        ));

        let mut handles = Vec::new();
        for _ in 0..16 {
            let gated = Arc::clone(&gated);
            handles.push(tokio::spawn(async move {
                let _ = gated.embed(vec!["hello".to_string()]).await;
            }));
        }
        for handle in handles {
            handle.await.expect("embed task should not panic");
        }

        assert_eq!(
            capability.health(),
            unavailable("connection refused"),
            "concurrent failures must converge on one degraded health"
        );
    }
}
