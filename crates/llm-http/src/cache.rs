//! Shared TTL cache for a connector's `list_models()` result.
//!
//! Issue #620. The OpenRouter / Azure / Google connectors each want the same
//! behaviour: serve the last model listing for a while instead of hitting the
//! (sometimes large) live catalogue on every model-picker open, and degrade to
//! their curated table when the live fetch fails. `llm-bedrock` already proved
//! the pattern with a `ModelClock` + 1h TTL; this module lifts it into one
//! reusable primitive so the three newer connectors don't each re-derive it.
//!
//! ## Threading and `.await`
//!
//! Connectors are shared as `Arc<dyn LlmClient>` across concurrent turns, so
//! the cache is `Send + Sync` — a [`std::sync::Mutex`] guards the single entry.
//! The guard is **never** held across an `.await`: [`ModelCache::cached`] clones
//! the snapshot out under the lock and drops the guard before returning, and
//! [`ModelCache::store`] writes under the lock and drops it. The caller performs
//! the (async) live fetch entirely outside the lock, so `clippy::await_holding_lock`
//! never has anything to fire on.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use desktop_assistant_core::ports::llm::ModelInfo;

/// Default TTL for a model-list cache. One hour is cheap to refresh and long
/// enough that a UI does not trigger a round-trip on every model-picker open.
pub const DEFAULT_MODEL_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

/// Abstraction over [`Instant::now`] so a cache-TTL test can advance time
/// deterministically instead of sleeping. The production impl is [`SystemClock`];
/// tests inject a mock that returns a controllable instant.
pub trait Clock: Send + Sync {
    /// The current monotonic instant.
    fn now(&self) -> Instant;
}

/// Default [`Clock`] reading the monotonic OS clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A stored snapshot: the models and the instant they were fetched.
struct CacheEntry {
    fetched_at: Instant,
    models: Vec<ModelInfo>,
}

/// A thread-safe TTL cache for a connector's `list_models()` result.
///
/// Construct with [`ModelCache::new`] (default TTL + [`SystemClock`]); override
/// the TTL with [`ModelCache::set_ttl`] and inject a test clock with
/// [`ModelCache::set_clock`]. Read with [`ModelCache::cached`] (a miss returns
/// `None`) and populate with [`ModelCache::store`].
pub struct ModelCache {
    entry: Mutex<Option<CacheEntry>>,
    ttl: Duration,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for ModelCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Report only whether an entry is present (never the model list) and
        // the TTL; the clock is opaque.
        let populated = self.entry.lock().map(|g| g.is_some()).unwrap_or(false);
        f.debug_struct("ModelCache")
            .field("populated", &populated)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelCache {
    /// A cache with the [`DEFAULT_MODEL_CACHE_TTL`] and a [`SystemClock`].
    pub fn new() -> Self {
        Self {
            entry: Mutex::new(None),
            ttl: DEFAULT_MODEL_CACHE_TTL,
            clock: Arc::new(SystemClock),
        }
    }

    /// Set the time-to-live applied to a cached listing. A cached entry older
    /// than this is treated as a miss.
    pub fn set_ttl(&mut self, ttl: Duration) {
        self.ttl = ttl;
    }

    /// Inject a [`Clock`] — the test seam for deterministic TTL expiry.
    pub fn set_clock(&mut self, clock: Arc<dyn Clock>) {
        self.clock = clock;
    }

    /// The configured TTL.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Return a clone of the cached listing when one is present and still within
    /// the TTL; `None` when the cache is empty or the entry has expired. The
    /// lock is released before returning, so the caller may `.await` a live
    /// fetch on a miss without holding it.
    ///
    /// Boundary: an entry whose age is exactly the TTL is a miss (`age < ttl`),
    /// so the caller re-fetches at the boundary rather than serving a just-stale
    /// listing.
    pub fn cached(&self) -> Option<Vec<ModelInfo>> {
        let guard = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        let entry = guard.as_ref()?;
        let age = self.clock.now().saturating_duration_since(entry.fetched_at);
        (age < self.ttl).then(|| entry.models.clone())
    }

    /// Replace the cached listing, stamping it with the current instant so its
    /// TTL runs from now. Call this only with a successfully fetched listing —
    /// storing a degraded/empty result would poison the cache and suppress the
    /// next live retry.
    pub fn store(&self, models: Vec<ModelInfo>) {
        let now = self.clock.now();
        let mut guard = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(CacheEntry {
            fetched_at: now,
            models,
        });
    }
}
