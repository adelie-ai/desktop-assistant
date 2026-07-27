//! Embedding-backend health: the startup probe, and the live capability it
//! seeds.
//!
//! `EmbeddingsSettingsView.available` is a shallow connector-string check, not
//! a probe: a misconfigured embedder (for example a text-generation model that
//! answers every embed with HTTP 501) reads as healthy and silently disables
//! all vector search behind a green status. [`probe_embedding_backend`]
//! performs one tiny embed at startup so a broken backend is caught,
//! classified, and surfaced as a real degraded-health state instead.
//!
//! The probe is model-agnostic: it catches *any* backend that cannot produce a
//! vector, regardless of the model's name. It is the primary safety net; the
//! name-based generation-model denylist in [`crate::config`] is only a faster,
//! clearer secondary guard for the common misconfiguration.
//!
//! # Live capability, not a boot-time latch
//!
//! Why: the probe's verdict is a snapshot of one moment, and the moment it
//! samples is daemon startup — exactly when a co-hosted Ollama is most likely to
//! be restarting or cold-loading a GGUF. Treating that snapshot as final made a
//! momentary outage permanent: vector search, the embedding backfill loop,
//! dreaming, consolidation and the whole knowledge-maintenance handler stayed
//! off until someone restarted the daemon.
//!
//! So the probe seeds an [`EmbeddingCapability`] — a shared, mutable
//! [`EmbeddingHealth`] — instead of deciding anything permanently:
//!
//! - the client stays wired whenever a backend is *configured*, so the
//!   background loops and the maintenance handler exist and can recover;
//! - [`HealthGatedEmbeddingClient`] gates every embed on the current health and
//!   feeds each outcome back into it, so a live failure degrades the capability
//!   and a live success restores it;
//! - [`recheck_degraded_backend`] re-probes on an interval *while degraded*, so
//!   a backend that came back is noticed with no traffic and no restart;
//! - the same handle backs `GetConfig`, so a client can see the current state
//!   and its reason rather than a stale boot verdict.
//!
//! That is this codebase's capability model applied to embeddings: "is the
//! capability present?" ([`EmbeddingHealth::Disabled`] vs the rest) is a
//! separate question from "did my call succeed?" ([`EmbeddingHealth::Ok`] vs
//! [`EmbeddingHealth::Unavailable`]), and the reason a feature is off is
//! surfaced rather than swallowed.

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

/// Reason recorded when the backend answers but produces no usable vector.
const NO_VECTORS_REASON: &str = "embedding backend returned no vectors";

/// Upper bound, in characters, on a stored [`EmbeddingHealth::Unavailable`]
/// reason. Backend errors quote the raw HTTP response body, which a base URL
/// pointing at the wrong service can make arbitrarily large, and the reason then
/// rides in every `GetConfig` payload for as long as the backend is degraded.
/// Bounded here, once, rather than at each producer.
const MAX_REASON_CHARS: usize = 512;

/// Bound the reason carried by an [`EmbeddingHealth`], on a character boundary
/// so multi-byte text cannot panic the truncation.
fn bounded(health: EmbeddingHealth) -> EmbeddingHealth {
    let EmbeddingHealth::Unavailable { reason } = health else {
        return health;
    };
    let Some((split, _)) = reason.char_indices().nth(MAX_REASON_CHARS) else {
        return EmbeddingHealth::Unavailable { reason };
    };
    EmbeddingHealth::Unavailable {
        reason: format!("{}... (truncated)", &reason[..split]),
    }
}

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
                    reason: NO_VECTORS_REASON.to_string(),
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
/// Only an *absent* backend ([`Disabled`](EmbeddingHealth::Disabled) — nothing
/// configured) drops it. A failed probe
/// ([`Unavailable`](EmbeddingHealth::Unavailable)) and an undetermined one
/// ([`Unknown`](EmbeddingHealth::Unknown)) both keep it, because the probe
/// samples exactly the moment a co-hosted backend is most likely to be
/// restarting or cold-loading; dropping the client there would latch off the
/// backfill loop, dreaming, consolidation and the knowledge-maintenance handler
/// for the whole process lifetime. Whether a *call* may be made right now is a
/// separate, live question answered by [`EmbeddingCapability`], which
/// [`HealthGatedEmbeddingClient`] enforces on every embed.
pub(crate) fn keep_embedding_client(health: &EmbeddingHealth) -> bool {
    !matches!(health, EmbeddingHealth::Disabled)
}

/// Whether the startup stale-embedding sweep may run.
///
/// The sweep is destructive — it NULLs every stored vector whose model stamp
/// differs from the current one — so it needs proof of both facts it rests on:
/// the backend really embeds (`health`), and the model identity came from the
/// backend rather than from the configured name as a fallback
/// (`model_resolved`). Keeping the client wired through a failed probe means
/// this decision can no longer ride on the client merely existing; a guessed
/// identity would wipe good vectors that only *look* stale.
pub(crate) fn sweep_stale_embeddings(model_resolved: bool, health: &EmbeddingHealth) -> bool {
    model_resolved && matches!(health, EmbeddingHealth::Ok)
}

/// The live health of the embedding backend, shared between the daemon's vector
/// paths, the background re-check loop and `GetConfig`.
///
/// Cheap to read (an `RwLock` read plus a clone of a small enum) and written
/// only on a *change* of state, so the hot embed path can consult it per call.
/// A poisoned lock is recovered rather than propagated: a panic elsewhere must
/// not turn health reporting into a second failure.
#[derive(Debug)]
pub struct EmbeddingCapability {
    health: RwLock<EmbeddingHealth>,
}

impl EmbeddingCapability {
    /// Seed the capability with the startup probe's verdict.
    pub fn new(initial: EmbeddingHealth) -> Self {
        Self {
            health: RwLock::new(bounded(initial)),
        }
    }

    /// The health as of now.
    pub fn health(&self) -> EmbeddingHealth {
        self.health
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// `Some(reason)` when the backend must not be called right now, carrying
    /// why so a caller can say what is wrong instead of failing opaquely.
    ///
    /// [`Unknown`](EmbeddingHealth::Unknown) is deliberately callable: a
    /// configured-but-undetermined backend is attempted optimistically, and the
    /// attempt itself settles the question.
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

    /// Store `next`, logging only genuine transitions so a backend that is down
    /// for an hour costs one line rather than one per embed.
    fn set(&self, next: EmbeddingHealth) {
        let next = bounded(next);
        let mut guard = self
            .health
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if *guard == next {
            return;
        }
        match &next {
            EmbeddingHealth::Ok => {
                tracing::info!("embedding backend healthy; vector search live")
            }
            EmbeddingHealth::Unavailable { reason } => tracing::warn!(
                "embedding backend unavailable; vector search degraded to full-text search: {reason}"
            ),
            EmbeddingHealth::Disabled | EmbeddingHealth::Unknown => {}
        }
        *guard = next;
    }
}

/// [`EmbeddingClient`] decorator that gates every embed on the live
/// [`EmbeddingCapability`] and reports each outcome back into it.
///
/// Gating and observing belong together: the gate is only honest if something
/// keeps the health current, and the cheapest source of truth is the traffic
/// already flowing through the client. While the backend is
/// [`Unavailable`](EmbeddingHealth::Unavailable) the call is declined without
/// touching the network, so query embedding, the backfill loop and dreaming take
/// their existing degraded routes instead of churning against a dead backend —
/// and [`recheck_degraded_backend`] is what lifts the gate again.
///
/// Trade-off: one failed embed closes the gate for everyone until the next
/// re-check, so a fault local to a single request costs up to
/// [`RECHECK_INTERVAL`] of full-text-only search. That is accepted rather than
/// debounced, because the backends this daemon drives fail as a unit (a
/// connection refused, a model not loaded, a non-embedding model answering 501)
/// and per-input rejection is already designed out upstream — the backfill asks
/// its backend to truncate oversized text rather than reject it. Distinguishing
/// the two would mean reading error strings, which this codebase does not do.
pub struct HealthGatedEmbeddingClient {
    inner: Arc<dyn EmbeddingClient>,
    capability: Arc<EmbeddingCapability>,
}

impl HealthGatedEmbeddingClient {
    /// Wrap `inner`, gating on and reporting into `capability`.
    pub fn new(inner: Arc<dyn EmbeddingClient>, capability: Arc<EmbeddingCapability>) -> Self {
        Self { inner, capability }
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for HealthGatedEmbeddingClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        if let Some(reason) = self.capability.unavailable_reason() {
            return Err(CoreError::Llm(format!(
                "embedding backend unavailable: {reason}"
            )));
        }

        // An empty batch legitimately yields no vectors, so it says nothing
        // about the backend's health and must not be read as a failure.
        let observable = !texts.is_empty();
        let result = self.inner.embed(texts).await;
        match &result {
            Ok(vectors) if observable => {
                if vectors.iter().any(|vector| !vector.is_empty()) {
                    self.capability.record_ok();
                } else {
                    // HTTP 200 with nothing usable: the call succeeded, the
                    // capability did not.
                    self.capability.record_failure(NO_VECTORS_REASON);
                }
            }
            Ok(_) => {}
            Err(error) => self.capability.record_failure(error.to_string()),
        }
        result
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        self.inner.model_identifier().await
    }
}

/// How often a *degraded* embedding backend is re-probed. Long enough that a
/// backend that is down for a while costs nothing much, short enough that a
/// restarting Ollama is picked up well within a user's patience.
pub const RECHECK_INTERVAL: Duration = Duration::from_secs(60);

/// Re-probe the backend while it reads
/// [`Unavailable`](EmbeddingHealth::Unavailable), so recovery needs neither
/// traffic nor a daemon restart.
///
/// Takes the *undecorated* client: the gate in [`HealthGatedEmbeddingClient`]
/// would otherwise decline exactly the call that could lift it. While the
/// capability reads healthy the loop makes no calls at all, so the happy path
/// costs one timer wake-up per `interval`.
///
/// Runs until `shutdown` fires or its sender is dropped, including *during* a
/// probe: a probe of an unresponsive backend can take
/// `PROBE_ATTEMPTS * PROBE_TIMEOUT`, which is far too long to make daemon
/// shutdown wait.
pub async fn recheck_degraded_backend(
    client: Arc<dyn EmbeddingClient>,
    capability: Arc<EmbeddingCapability>,
    interval: Duration,
    shutdown: oneshot::Receiver<()>,
) {
    let mut shutdown = shutdown;
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = &mut shutdown => {
                tracing::debug!("embedding health re-check: shutdown signal received");
                return;
            }
        }

        if !matches!(capability.health(), EmbeddingHealth::Unavailable { .. }) {
            continue;
        }

        let health = tokio::select! {
            health = probe_embedding_backend(client.as_ref()) => health,
            _ = &mut shutdown => {
                tracing::debug!("embedding health re-check: shutdown signal received mid-probe");
                return;
            }
        };
        match health {
            EmbeddingHealth::Ok => capability.record_ok(),
            EmbeddingHealth::Unavailable { reason } => capability.record_failure(reason),
            // The probe never returns these; the caller sets them when there is
            // no backend to probe.
            EmbeddingHealth::Disabled | EmbeddingHealth::Unknown => {}
        }
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
    fn stale_embedding_sweep_needs_a_healthy_backend_and_a_resolved_model() {
        // Keeping the client wired through a failed probe must not let the
        // destructive startup sweep run on a guessed model identity: a backend
        // that was down at boot leaves the configured name (no digest) as the
        // "current" model, and sweeping on that wipes every good vector.
        assert!(!sweep_stale_embeddings(
            false,
            &EmbeddingHealth::Unavailable {
                reason: "connection refused".to_string(),
            }
        ));
        assert!(
            !sweep_stale_embeddings(false, &EmbeddingHealth::Ok),
            "an unresolved model identity must not drive the sweep"
        );
        assert!(
            !sweep_stale_embeddings(
                true,
                &EmbeddingHealth::Unavailable {
                    reason: "HTTP 501".to_string(),
                }
            ),
            "a backend that cannot embed must not drive the sweep"
        );
        assert!(!sweep_stale_embeddings(true, &EmbeddingHealth::Unknown));
        assert!(!sweep_stale_embeddings(true, &EmbeddingHealth::Disabled));
        assert!(sweep_stale_embeddings(true, &EmbeddingHealth::Ok));
    }

    #[test]
    fn absent_embedding_backend_still_drops_the_client() {
        // Absent is not degraded: with no backend configured at all there is
        // nothing to wire and nothing to re-probe.
        assert!(!keep_embedding_client(&EmbeddingHealth::Disabled));
    }

    // --- Live capability -------------------------------------------------

    /// Poll `capability` until its health satisfies `wanted`, bounded so a
    /// regression fails the test rather than hanging the suite.
    async fn await_health(
        capability: &EmbeddingCapability,
        wanted: impl Fn(&EmbeddingHealth) -> bool,
    ) -> bool {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !wanted(&capability.health()) {
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
            await_health(&capability, |health| *health == EmbeddingHealth::Ok).await,
            "a backend that came back must re-probe healthy without a restart, health was {:?}",
            capability.health()
        );

        let _ = shutdown_tx.send(());
        task.await.expect("recheck task should exit cleanly");
    }

    #[tokio::test]
    async fn recheck_keeps_a_still_broken_backend_degraded() {
        // Recovery must be evidence-based: a backend that is still refusing
        // stays Unavailable, and the re-check replaces the stale boot reason
        // with what the backend says now.
        let capability = Arc::new(EmbeddingCapability::new(unavailable(
            "embedding probe timed out",
        )));
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
            await_health(&capability, |health| matches!(
                health,
                EmbeddingHealth::Unavailable { reason } if reason.contains("501")
            ))
            .await,
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

    #[test]
    fn recorded_failure_reason_is_bounded() {
        // Malformed backend: a base URL pointing at the wrong service answers
        // with a whole document, and that body becomes the error text. The
        // stored reason must stay a bounded, safely-split string — it is
        // surfaced in every `GetConfig` payload while the backend is degraded.
        let capability = EmbeddingCapability::new(EmbeddingHealth::Ok);
        // Multi-byte throughout, so a byte-wise cut would split a character.
        capability.record_failure("é".repeat(MAX_REASON_CHARS * 4));
        match capability.health() {
            EmbeddingHealth::Unavailable { reason } => {
                assert!(
                    reason.chars().count() <= MAX_REASON_CHARS + "... (truncated)".len(),
                    "reason must be bounded, got {} chars",
                    reason.chars().count()
                );
                assert!(reason.ends_with("... (truncated)"));
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn short_failure_reason_is_kept_verbatim() {
        // Boundary: a reason at the cap is stored untouched, so the common case
        // never gains a misleading "truncated" marker.
        let capability = EmbeddingCapability::new(EmbeddingHealth::Ok);
        let reason = "x".repeat(MAX_REASON_CHARS);
        capability.record_failure(reason.clone());
        assert_eq!(
            capability.health(),
            EmbeddingHealth::Unavailable { reason },
            "a reason at the cap must be kept verbatim"
        );
    }

    #[tokio::test]
    async fn recheck_stops_on_shutdown_mid_probe() {
        // Worst case for shutdown: the backend accepted the connection and then
        // went silent, so the probe is inside its multi-attempt timeout budget.
        // Shutdown must still be prompt rather than waiting minutes for a
        // backend that is never going to answer.
        let capability = Arc::new(EmbeddingCapability::new(unavailable("down")));
        let client: Arc<dyn EmbeddingClient> = Arc::new(MockEmbedder {
            fail_reason: None,
            vectors: vec![vec![0.1]],
            cold_delay: Duration::from_secs(3600),
            slow_calls: usize::MAX,
            ..Default::default()
        });
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(recheck_degraded_backend(
            client,
            Arc::clone(&capability),
            Duration::from_millis(5),
            shutdown_rx,
        ));

        // Let the loop get into the probe before asking it to stop.
        tokio::time::sleep(Duration::from_millis(30)).await;
        let _ = shutdown_tx.send(());
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("recheck must stop promptly even mid-probe")
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

        match capability.health() {
            EmbeddingHealth::Unavailable { reason } => assert!(
                reason.contains("connection refused"),
                "concurrent failures must converge on one degraded health, got: {reason}"
            ),
            other => panic!("concurrent failures must degrade the health, got {other:?}"),
        }
    }
}
