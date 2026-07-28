//! Failure throttling for `POST /login` (#808).
//!
//! The login door mints a bearer token from a username and a password. Without
//! a limit it answers guesses at full request rate, which makes it a password
//! oracle: on a bare-metal deploy the credential being guessed is a real OS
//! account password, checked through PAM.
//!
//! Two counters, because there are two attack shapes and each is blind to the
//! other:
//!
//! - **per source address** - one caller walking a list of usernames;
//! - **per username** - many callers converging on one account.
//!
//! A failure increments both. Either being locked refuses the request, and one
//! success clears both, so an ordinary mistyped password costs nothing lasting.
//! The lockout starts at [`BASE_LOCKOUT`] and doubles with each further
//! failure, up to [`MAX_LOCKOUT`]: legitimate retries stay cheap, sustained
//! guessing does not.
//!
//! Refusals do **not** count. A caller polling a locked door would otherwise
//! extend its own lockout for ever, which turns a mistyped password into a
//! permanent outage for the person who made it.
//!
//! The caller supplies `now` rather than the module reading the clock, so the
//! policy is testable without sleeping and without a paused runtime.
//!
//! ## Scope
//!
//! This is a per-process, in-memory limit on one endpoint. It is not a
//! distributed rate limiter and does not survive a restart; both are
//! deliberate, because the design record puts the tenant boundary inside one
//! organization rather than at hostile isolation. It bounds an unauthenticated
//! guessing loop, which the door had nothing against at all.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Failures a counter absorbs before the door starts refusing. Enough for a
/// person to mistype a password a few times and still get in.
const FREE_ATTEMPTS: u32 = 5;

/// The lockout for the first failure past the budget. Each further failure
/// doubles it.
const BASE_LOCKOUT: Duration = Duration::from_secs(30);

/// The ceiling the doubling stops at, so a counter cannot lock a legitimate
/// caller out for a day.
const MAX_LOCKOUT: Duration = Duration::from_secs(15 * 60);

/// A counter with no failure for this long starts again from zero. Slow
/// guessing is still caught, because the counter only resets after a quiet
/// window, not on a timer.
const IDLE_RESET: Duration = Duration::from_secs(15 * 60);

/// Upper bound on tracked counters. The keys are attacker-supplied (any source
/// address, any username), so the map needs a ceiling or the throttle becomes
/// its own memory-exhaustion path. At the cap the least recently active counter
/// is dropped.
const MAX_TRACKED: usize = 4096;

/// Longest username kept as a counter key. A caller cannot grow the map with
/// long strings, and no real login name is anywhere near this.
const MAX_KEY_CHARS: usize = 64;

/// One of the two counters a failed attempt increments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Counter {
    /// The connecting peer's address. Absent when the server was not wired to
    /// report it, in which case only the username counter applies.
    Source(IpAddr),
    Username(String),
}

/// A counter's state.
#[derive(Debug)]
struct Attempts {
    failures: u32,
    last_failure: Instant,
    locked_until: Option<Instant>,
}

/// The `/login` failure counters for one server.
#[derive(Debug, Default)]
pub(crate) struct LoginThrottle {
    counters: Mutex<HashMap<Counter, Attempts>>,
}

impl LoginThrottle {
    /// How much longer this caller is refused, or `None` if the door is open.
    pub(crate) fn locked_for(
        &self,
        source: Option<IpAddr>,
        username: &str,
        now: Instant,
    ) -> Option<Duration> {
        let counters = self.counters();
        keys(source, username)
            .into_iter()
            .filter_map(|key| counters.get(&key)?.locked_until)
            .map(|until| until.saturating_duration_since(now))
            .filter(|left| !left.is_zero())
            .max()
    }

    /// Count one failed attempt against both counters. Returns the lockout the
    /// caller has just earned, if any, so the handler can say so once.
    pub(crate) fn record_failure(
        &self,
        source: Option<IpAddr>,
        username: &str,
        now: Instant,
    ) -> Option<Duration> {
        let mut counters = self.counters();
        make_room(&mut counters, now);

        let mut longest: Option<Duration> = None;
        for key in keys(source, username) {
            let entry = counters.entry(key).or_insert(Attempts {
                failures: 0,
                last_failure: now,
                locked_until: None,
            });
            if now.saturating_duration_since(entry.last_failure) >= IDLE_RESET {
                entry.failures = 0;
                entry.locked_until = None;
            }
            entry.failures = entry.failures.saturating_add(1);
            entry.last_failure = now;
            entry.locked_until =
                lockout_for(entry.failures).map(|lockout| now.checked_add(lockout).unwrap_or(now));
            longest = longest.max(
                entry
                    .locked_until
                    .map(|until| until.saturating_duration_since(now)),
            );
        }
        longest
    }

    /// Clear both counters after a successful login.
    pub(crate) fn record_success(&self, source: Option<IpAddr>, username: &str) {
        let mut counters = self.counters();
        for key in keys(source, username) {
            counters.remove(&key);
        }
    }

    /// Take the lock, recovering from a poisoned one rather than propagating
    /// the panic. A previous panic while holding it must not turn the login
    /// door into a permanent outage; the map is a cache of counters, so the
    /// worst case of using it after a panic is one miscounted attempt.
    fn counters(&self) -> std::sync::MutexGuard<'_, HashMap<Counter, Attempts>> {
        self.counters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The counters one attempt touches.
fn keys(source: Option<IpAddr>, username: &str) -> Vec<Counter> {
    let mut keys = Vec::with_capacity(2);
    if let Some(address) = source {
        keys.push(Counter::Source(address));
    }
    keys.push(Counter::Username(
        username.chars().take(MAX_KEY_CHARS).collect(),
    ));
    keys
}

/// The lockout `failures` earns, or `None` while the budget is unspent.
fn lockout_for(failures: u32) -> Option<Duration> {
    let over_budget = failures.checked_sub(FREE_ATTEMPTS)?;
    if over_budget == 0 {
        return None;
    }
    // The first failure past the budget doubles nothing.
    let doublings = over_budget - 1;
    let factor = 1u32.checked_shl(doublings).unwrap_or(u32::MAX);
    Some(BASE_LOCKOUT.saturating_mul(factor).min(MAX_LOCKOUT))
}

/// Drop counters that have gone quiet, then - if the map is still at its cap -
/// the least recently active ones, so there is room for the two this attempt
/// will insert.
fn make_room(counters: &mut HashMap<Counter, Attempts>, now: Instant) {
    counters.retain(|_, entry| now.saturating_duration_since(entry.last_failure) < IDLE_RESET);
    while counters.len() + 2 > MAX_TRACKED {
        let Some(oldest) = counters
            .iter()
            .min_by_key(|(_, entry)| entry.last_failure)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        counters.remove(&oldest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALICE: &str = "alice";

    fn source() -> Option<IpAddr> {
        Some(IpAddr::from([192, 0, 2, 10]))
    }

    fn other_source() -> Option<IpAddr> {
        Some(IpAddr::from([198, 51, 100, 7]))
    }

    /// Spend the whole budget plus `extra` failures.
    fn fail(
        throttle: &LoginThrottle,
        from: Option<IpAddr>,
        username: &str,
        times: u32,
        now: Instant,
    ) {
        for _ in 0..times {
            throttle.record_failure(from, username, now);
        }
    }

    #[test]
    fn the_budget_is_spent_before_the_door_closes() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        fail(&throttle, source(), ALICE, FREE_ATTEMPTS, now);
        assert_eq!(
            throttle.locked_for(source(), ALICE, now),
            None,
            "a person mistyping a password a few times must still get in"
        );

        throttle.record_failure(source(), ALICE, now);
        assert_eq!(
            throttle.locked_for(source(), ALICE, now),
            Some(BASE_LOCKOUT),
            "the first failure past the budget earns the base lockout"
        );
    }

    #[test]
    fn the_lockout_doubles_and_stops_at_the_ceiling() {
        assert_eq!(lockout_for(FREE_ATTEMPTS), None);
        assert_eq!(lockout_for(FREE_ATTEMPTS + 1), Some(BASE_LOCKOUT));
        assert_eq!(lockout_for(FREE_ATTEMPTS + 2), Some(BASE_LOCKOUT * 2));
        assert_eq!(lockout_for(FREE_ATTEMPTS + 3), Some(BASE_LOCKOUT * 4));
        assert_eq!(lockout_for(FREE_ATTEMPTS + 100), Some(MAX_LOCKOUT));
        assert_eq!(
            lockout_for(u32::MAX),
            Some(MAX_LOCKOUT),
            "the doubling must not overflow at an absurd failure count"
        );
    }

    #[test]
    fn the_lockout_expires() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        fail(&throttle, source(), ALICE, FREE_ATTEMPTS + 1, now);

        let still_locked = now + BASE_LOCKOUT - Duration::from_millis(1);
        assert!(throttle.locked_for(source(), ALICE, still_locked).is_some());

        let expired = now + BASE_LOCKOUT;
        assert_eq!(
            throttle.locked_for(source(), ALICE, expired),
            None,
            "the door reopens once the lockout has run out"
        );
    }

    #[test]
    fn a_success_clears_both_counters() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        fail(&throttle, source(), ALICE, FREE_ATTEMPTS + 1, now);
        assert!(throttle.locked_for(source(), ALICE, now).is_some());

        throttle.record_success(source(), ALICE);
        assert_eq!(throttle.locked_for(source(), ALICE, now), None);
        assert_eq!(
            throttle.record_failure(source(), ALICE, now),
            None,
            "the counters restarted, so the next failure is inside the budget again"
        );
    }

    /// One source walking a username list is caught by the source counter even
    /// though no single username reaches the budget.
    #[test]
    fn the_source_counter_adds_up_across_usernames() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        for index in 0..=FREE_ATTEMPTS {
            throttle.record_failure(source(), &format!("user-{index}"), now);
        }
        assert!(
            throttle.locked_for(source(), "someone-new", now).is_some(),
            "the source must be locked out however many usernames it tried"
        );
    }

    /// Many sources converging on one account are caught by the username
    /// counter, even though no single source reaches the budget.
    #[test]
    fn the_username_counter_adds_up_across_sources() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        for index in 0..=FREE_ATTEMPTS {
            let address = IpAddr::from([203, 0, 113, index as u8]);
            throttle.record_failure(Some(address), ALICE, now);
        }
        assert!(
            throttle
                .locked_for(Some(IpAddr::from([203, 0, 113, 200])), ALICE, now)
                .is_some(),
            "the account must be protected however many addresses tried it"
        );
    }

    /// One caller's failures must not lock a different caller out of a
    /// different account.
    #[test]
    fn an_unrelated_caller_is_not_locked_out() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        fail(&throttle, source(), ALICE, FREE_ATTEMPTS + 4, now);
        assert_eq!(throttle.locked_for(other_source(), "bob", now), None);
    }

    /// With no source address the username counter still applies, so a server
    /// that cannot report the peer address is throttled rather than unlimited.
    #[test]
    fn a_missing_source_address_still_counts_the_username() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        fail(&throttle, None, ALICE, FREE_ATTEMPTS + 1, now);
        assert!(throttle.locked_for(None, ALICE, now).is_some());
    }

    /// A quiet window forgives the counter, so a mistyped password months ago
    /// does not add to today's.
    #[test]
    fn a_quiet_counter_starts_again() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        fail(&throttle, source(), ALICE, FREE_ATTEMPTS, now);

        let later = now + IDLE_RESET;
        assert_eq!(
            throttle.record_failure(source(), ALICE, later),
            None,
            "after a quiet window the counter starts from zero"
        );
    }

    /// The keys are attacker-supplied, so the map must not grow without bound.
    #[test]
    fn the_counter_map_is_bounded() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        for index in 0..(MAX_TRACKED * 2) {
            throttle.record_failure(None, &format!("user-{index}"), now);
        }
        let tracked = throttle.counters().len();
        assert!(
            tracked <= MAX_TRACKED,
            "the throttle must not become its own memory-exhaustion path, tracked {tracked}"
        );
    }

    /// A long username cannot be used to grow the map entry by entry.
    #[test]
    fn a_username_key_is_truncated() {
        let long = "x".repeat(MAX_KEY_CHARS * 4);
        let keys = keys(None, &long);
        match keys.as_slice() {
            [Counter::Username(name)] => assert_eq!(name.chars().count(), MAX_KEY_CHARS),
            other => panic!("expected one username counter, got {other:?}"),
        }
    }
}
