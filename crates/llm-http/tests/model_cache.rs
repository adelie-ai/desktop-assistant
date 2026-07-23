//! Unit tests for the shared TTL model-list cache (issue #620).
//!
//! Time is injected via a mock [`Clock`] so the TTL logic is exercised
//! deterministically — no sleeping, no real wall-clock dependency.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use desktop_assistant_core::ports::llm::ModelInfo;
use desktop_assistant_llm_http::{Clock, DEFAULT_MODEL_CACHE_TTL, ModelCache};

/// A clock backed by an atomic second-offset from a fixed origin. Tests drive
/// it forward with [`MockClock::advance_secs`]; it never reads the OS clock
/// after construction.
struct MockClock {
    origin: Instant,
    offset_secs: AtomicU64,
}

impl MockClock {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            origin: Instant::now(),
            offset_secs: AtomicU64::new(0),
        })
    }

    fn advance_secs(&self, secs: u64) {
        self.offset_secs.fetch_add(secs, Ordering::SeqCst);
    }
}

impl Clock for MockClock {
    fn now(&self) -> Instant {
        self.origin + Duration::from_secs(self.offset_secs.load(Ordering::SeqCst))
    }
}

fn cache_with(clock: Arc<MockClock>, ttl: Duration) -> ModelCache {
    let mut cache = ModelCache::new();
    cache.set_clock(clock);
    cache.set_ttl(ttl);
    cache
}

fn models(ids: &[&str]) -> Vec<ModelInfo> {
    ids.iter().map(|id| ModelInfo::new(*id)).collect()
}

#[test]
fn default_ttl_is_one_hour() {
    assert_eq!(DEFAULT_MODEL_CACHE_TTL, Duration::from_secs(60 * 60));
    // A freshly constructed cache adopts the default TTL.
    assert_eq!(ModelCache::new().ttl(), DEFAULT_MODEL_CACHE_TTL);
}

#[test]
fn cached_is_none_when_empty() {
    let clock = MockClock::new();
    let cache = cache_with(clock, Duration::from_secs(3600));
    assert!(cache.cached().is_none(), "empty cache must miss");
}

#[test]
fn store_then_cached_returns_within_ttl() {
    let clock = MockClock::new();
    let cache = cache_with(clock.clone(), Duration::from_secs(3600));
    let m = models(&["a", "b"]);
    cache.store(m.clone());

    // No time has passed → served from cache.
    assert_eq!(cache.cached(), Some(m.clone()));
    // Still within the TTL after a partial advance.
    clock.advance_secs(3599);
    assert_eq!(cache.cached(), Some(m));
}

#[test]
fn cached_is_none_after_ttl_expiry() {
    let clock = MockClock::new();
    let cache = cache_with(clock.clone(), Duration::from_secs(3600));
    cache.store(models(&["stale"]));
    clock.advance_secs(3601);
    assert!(
        cache.cached().is_none(),
        "entry older than the TTL must be treated as a miss"
    );
}

#[test]
fn cached_at_exactly_ttl_is_expired() {
    // Boundary: age == TTL is expired (the check is `age < ttl`), so the caller
    // re-fetches exactly at the boundary rather than serving a just-stale entry.
    let clock = MockClock::new();
    let cache = cache_with(clock.clone(), Duration::from_secs(3600));
    cache.store(models(&["edge"]));
    clock.advance_secs(3600);
    assert!(
        cache.cached().is_none(),
        "age exactly equal to the TTL must be a miss"
    );
}

#[test]
fn cached_just_before_ttl_is_a_hit() {
    let clock = MockClock::new();
    let cache = cache_with(clock.clone(), Duration::from_secs(3600));
    let m = models(&["fresh"]);
    cache.store(m.clone());
    clock.advance_secs(3599);
    assert_eq!(
        cache.cached(),
        Some(m),
        "one second before the TTL is a hit"
    );
}

#[test]
fn store_restamps_freshness() {
    let clock = MockClock::new();
    let cache = cache_with(clock.clone(), Duration::from_secs(3600));
    cache.store(models(&["first"]));

    // Advance most of the way to expiry, then overwrite: the new entry is
    // stamped at the current instant, so it lives a full TTL from now.
    clock.advance_secs(3500);
    let second = models(&["second"]);
    cache.store(second.clone());

    // A further advance that would have expired the first entry (3500+200 >
    // 3600) still serves the second because it was restamped at t=3500.
    clock.advance_secs(200);
    assert_eq!(cache.cached(), Some(second));
}

#[test]
fn store_overwrites_previous_contents() {
    let clock = MockClock::new();
    let cache = cache_with(clock, Duration::from_secs(3600));
    cache.store(models(&["old"]));
    cache.store(models(&["new-a", "new-b"]));
    assert_eq!(cache.cached(), Some(models(&["new-a", "new-b"])));
}

#[test]
fn zero_ttl_never_serves_from_cache() {
    // A zero TTL means every entry is immediately stale (age 0 is not < 0).
    let clock = MockClock::new();
    let cache = cache_with(clock, Duration::from_secs(0));
    cache.store(models(&["x"]));
    assert!(cache.cached().is_none(), "a zero TTL can never hit");
}
