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
//! An attempt spends from both. Either being locked refuses the request, and
//! one success clears both, so an ordinary mistyped password costs nothing
//! lasting. The lockout starts at [`BASE_LOCKOUT`] and doubles with each
//! further failure, up to [`MAX_LOCKOUT`].
//!
//! ## Why the budget is spent before the password is checked
//!
//! [`LoginThrottle::begin_attempt`] checks the lockout **and** counts the
//! attempt under one lock, and the handler calls it before it validates the
//! credential. Checking first and counting afterwards is a check-then-act race
//! with an `.await` in the middle: a caller that opens many connections at once
//! has every one of them pass a check against counters that nothing has
//! recorded yet, so one budget of five buys as many guesses as the attacker can
//! open sockets. The password check is the slow half - a PAM call parks for the
//! whole of libpam's fail delay - which is exactly the window that race needs.
//! Counting first closes it: a request that got a slot is a request that paid
//! for it. A correct password then clears the counters, so nothing is spent in
//! the ordinary case.
//!
//! Refused requests do **not** count. A caller polling a locked door would
//! otherwise extend its own lockout for ever, which turns a mistyped password
//! into a permanent outage for the person who made it.
//!
//! ## Why the lockout stays short
//!
//! [`MAX_LOCKOUT`] is a minute, not an hour. Every deployment authenticates one
//! account and its name is not a secret, so a long lockout is a denial of
//! service anyone on the network can aim at the operator's own door: the
//! legitimate password is refused before it is even read. A minute keeps the
//! sustained guess rate at roughly one per minute - useless against any real
//! password - while the worst an attacker can cost a legitimate user is one
//! minute of waiting.
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
//!
//! It also does not read `X-Forwarded-For` or any other forwarding header,
//! because a caller writes those. Behind a reverse proxy every request
//! therefore carries the proxy's address and the per-source counter collapses
//! into one bucket; the per-username counter is unaffected.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Attempts a counter absorbs before the door starts refusing. Enough for a
/// person to mistype a password a few times and still get in.
const FREE_ATTEMPTS: u32 = 5;

/// The lockout for the first failure past the budget. Each further failure
/// doubles it.
const BASE_LOCKOUT: Duration = Duration::from_secs(5);

/// The ceiling the doubling stops at. Short on purpose - see "Why the lockout
/// stays short" in the module docs.
const MAX_LOCKOUT: Duration = Duration::from_secs(60);

/// A counter with no attempt for this long starts again from zero. Slow
/// guessing is still caught, because the counter only resets after a quiet
/// window, not on a timer. Comfortably longer than [`MAX_LOCKOUT`], so a
/// counter can never reopen and reset in the same instant.
const IDLE_RESET: Duration = Duration::from_secs(15 * 60);

/// Upper bound on tracked counters. The keys are attacker-supplied (any source
/// address, any username), so the map needs a ceiling or the throttle becomes
/// its own memory-exhaustion path. At the cap the least recently active counter
/// is dropped, and a counter serving a live lockout is kept in preference to
/// one that is not.
const MAX_TRACKED: usize = 4096;

/// Longest username kept as a counter key. A caller cannot grow the map with
/// long strings, and no real login name is anywhere near this. Log lines use
/// the same limit, through [`short_name`].
pub(crate) const MAX_KEY_CHARS: usize = 64;

/// Prefix length a source address is counted by.
///
/// An IPv4 address is counted whole. An IPv6 address is counted by its /64,
/// because a routed /64 is the standard allocation for one subscriber: keyed by
/// the full /128 an attacker rotates through 2^64 addresses and the per-source
/// counter never sees the same key twice.
const IPV6_COUNTED_BYTES: usize = 8;

/// The username, cut to the length the counter key uses, for a log line.
pub(crate) fn short_name(username: &str) -> String {
    username.chars().take(MAX_KEY_CHARS).collect()
}

/// One of the two counters an attempt spends from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Counter {
    /// The connecting peer's address, narrowed to [`IPV6_COUNTED_BYTES`] for
    /// IPv6. Absent when the server was not wired to report it, in which case
    /// only the username counter applies.
    Source(IpAddr),
    Username(String),
}

/// What [`LoginThrottle::begin_attempt`] decided.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Attempt {
    /// The credential may be checked. `earned_lockout` is the lockout this
    /// attempt has just put in force, if any, so the handler can report the
    /// transition once instead of on every later refusal.
    Allowed { earned_lockout: Option<Duration> },
    /// Refused without checking the credential. Try again after this long.
    Refused { retry_after: Duration },
}

/// A counter's state.
#[derive(Debug)]
struct Attempts {
    attempts: u32,
    last_attempt: Instant,
    locked_until: Option<Instant>,
}

impl Attempts {
    /// Whether this counter is currently refusing requests. A counter that is
    /// must survive the sweep in [`make_room`], or evicting it would release a
    /// lockout that has not expired.
    fn is_locked(&self, now: Instant) -> bool {
        self.locked_until.is_some_and(|until| until > now)
    }
}

/// The `/login` attempt counters for one server.
#[derive(Debug, Default)]
pub(crate) struct LoginThrottle {
    counters: Mutex<HashMap<Counter, Attempts>>,
    /// Whether the "no source address, so the per-source counter is off" notice
    /// has already been given. It is a property of how the server was started,
    /// so it is worth saying once and never again.
    missing_source_reported: AtomicBool,
}

impl LoginThrottle {
    /// Decide whether this attempt may proceed, and spend its budget if it may.
    ///
    /// Check and count are one operation under one lock. See the module docs
    /// for why the budget is spent before the credential is checked.
    pub(crate) fn begin_attempt(
        &self,
        source: Option<IpAddr>,
        username: &str,
        now: Instant,
    ) -> Attempt {
        let mut counters = self.counters();

        let refused = keys(source, username)
            .into_iter()
            .filter_map(|key| counters.get(&key)?.locked_until)
            .map(|until| until.saturating_duration_since(now))
            .filter(|left| !left.is_zero())
            .max();
        if let Some(retry_after) = refused {
            return Attempt::Refused { retry_after };
        }

        make_room(&mut counters, now);
        let mut earned_lockout: Option<Duration> = None;
        for key in keys(source, username) {
            let entry = counters.entry(key).or_insert(Attempts {
                attempts: 0,
                last_attempt: now,
                locked_until: None,
            });
            if now.saturating_duration_since(entry.last_attempt) >= IDLE_RESET {
                entry.attempts = 0;
                entry.locked_until = None;
            }
            entry.attempts = entry.attempts.saturating_add(1);
            entry.last_attempt = now;
            entry.locked_until =
                lockout_for(entry.attempts).map(|lockout| now.checked_add(lockout).unwrap_or(now));
            earned_lockout = earned_lockout.max(
                entry
                    .locked_until
                    .map(|until| until.saturating_duration_since(now)),
            );
        }
        Attempt::Allowed { earned_lockout }
    }

    /// Clear both counters after a successful login, so a person who mistyped
    /// their password a few times pays nothing lasting for it.
    pub(crate) fn record_success(&self, source: Option<IpAddr>, username: &str) {
        let mut counters = self.counters();
        for key in keys(source, username) {
            counters.remove(&key);
        }
    }

    /// Report, once, that this server does not supply a source address, so only
    /// the per-username counter is in force. Returns `true` the first time.
    ///
    /// The capability is either present for the life of the process or absent
    /// for it, so repeating the notice would only be noise.
    pub(crate) fn note_missing_source(&self) -> bool {
        !self.missing_source_reported.swap(true, Ordering::Relaxed)
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

/// The counters one attempt spends from.
fn keys(source: Option<IpAddr>, username: &str) -> Vec<Counter> {
    let mut keys = Vec::with_capacity(2);
    if let Some(address) = source {
        keys.push(Counter::Source(counted_address(address)));
    }
    keys.push(Counter::Username(short_name(username)));
    keys
}

/// Narrow a source address to the prefix it is counted by.
///
/// An IPv4-mapped address is unmapped first. A listener on `[::]` reports every
/// IPv4 peer as `::ffff:a.b.c.d`, and all of those share the same /64 (`::`), so
/// counting them as IPv6 would put every IPv4 caller in one bucket: five wrong
/// passwords from anywhere would lock out everyone else.
fn counted_address(address: IpAddr) -> IpAddr {
    match address.to_canonical() {
        IpAddr::V4(v4) => IpAddr::V4(v4),
        IpAddr::V6(v6) => {
            let mut octets = v6.octets();
            octets[IPV6_COUNTED_BYTES..].fill(0);
            IpAddr::V6(std::net::Ipv6Addr::from(octets))
        }
    }
}

/// The lockout in force once `attempts` have been spent, or `None` while the
/// budget still has room.
///
/// The budget is spent up front, so the attempt that reaches [`FREE_ATTEMPTS`]
/// is the last one allowed: it arms the base lockout, which refuses the next
/// one. Each further attempt doubles it.
fn lockout_for(attempts: u32) -> Option<Duration> {
    let doublings = attempts.checked_sub(FREE_ATTEMPTS)?;
    let factor = 1u32.checked_shl(doublings).unwrap_or(u32::MAX);
    Some(BASE_LOCKOUT.saturating_mul(factor).min(MAX_LOCKOUT))
}

/// Drop counters that have gone quiet, then - if the map is still at its cap -
/// the least recently active ones, so there is room for the two this attempt
/// will insert.
///
/// A counter serving a live lockout is never dropped by the quiet sweep, and is
/// the last to go at the cap. Dropping one would release a lockout early, which
/// is the one thing eviction must not be able to do.
fn make_room(counters: &mut HashMap<Counter, Attempts>, now: Instant) {
    counters.retain(|_, entry| {
        entry.is_locked(now) || now.saturating_duration_since(entry.last_attempt) < IDLE_RESET
    });
    while counters.len() + 2 > MAX_TRACKED {
        let Some(oldest) = counters
            .iter()
            .min_by_key(|(_, entry)| (entry.is_locked(now), entry.last_attempt))
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

    /// The lockout this attempt earned, or `None` when it was allowed with the
    /// budget still unspent. Panics if the attempt was refused.
    fn allow(
        throttle: &LoginThrottle,
        from: Option<IpAddr>,
        username: &str,
        now: Instant,
    ) -> Option<Duration> {
        match throttle.begin_attempt(from, username, now) {
            Attempt::Allowed { earned_lockout } => earned_lockout,
            Attempt::Refused { retry_after } => {
                panic!("expected the attempt to be allowed, refused for {retry_after:?}")
            }
        }
    }

    /// How long this attempt was refused for, or `None` if it was allowed.
    fn refusal(
        throttle: &LoginThrottle,
        from: Option<IpAddr>,
        username: &str,
        now: Instant,
    ) -> Option<Duration> {
        match throttle.begin_attempt(from, username, now) {
            Attempt::Refused { retry_after } => Some(retry_after),
            Attempt::Allowed { .. } => None,
        }
    }

    fn spend(
        throttle: &LoginThrottle,
        from: Option<IpAddr>,
        username: &str,
        times: u32,
        now: Instant,
    ) {
        for _ in 0..times {
            throttle.begin_attempt(from, username, now);
        }
    }

    #[test]
    fn the_budget_is_spent_before_the_door_closes() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        for attempt in 1..FREE_ATTEMPTS {
            assert_eq!(
                allow(&throttle, source(), ALICE, now),
                None,
                "attempt {attempt} is inside the budget"
            );
        }

        assert_eq!(
            allow(&throttle, source(), ALICE, now),
            Some(BASE_LOCKOUT),
            "the attempt that spends the budget earns the base lockout"
        );
        assert_eq!(
            refusal(&throttle, source(), ALICE, now),
            Some(BASE_LOCKOUT),
            "the next attempt is refused without checking a password"
        );
    }

    /// The property the concurrency race broke: an attempt that is refused has
    /// not spent anything, so a caller cannot lengthen its own lockout by
    /// polling, and cannot get more guesses than the budget by racing.
    #[test]
    fn a_refused_attempt_spends_nothing() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, source(), ALICE, FREE_ATTEMPTS, now);
        for _ in 0..100 {
            assert_eq!(refusal(&throttle, source(), ALICE, now), Some(BASE_LOCKOUT));
        }
        assert_eq!(
            refusal(&throttle, source(), ALICE, now),
            Some(BASE_LOCKOUT),
            "polling a locked door must not escalate the lockout"
        );
    }

    #[test]
    fn the_lockout_doubles_and_stops_at_the_ceiling() {
        assert_eq!(lockout_for(FREE_ATTEMPTS - 1), None);
        assert_eq!(lockout_for(FREE_ATTEMPTS), Some(BASE_LOCKOUT));
        assert_eq!(lockout_for(FREE_ATTEMPTS + 1), Some(BASE_LOCKOUT * 2));
        assert_eq!(lockout_for(FREE_ATTEMPTS + 2), Some(BASE_LOCKOUT * 4));
        assert_eq!(lockout_for(FREE_ATTEMPTS + 100), Some(MAX_LOCKOUT));
        assert_eq!(
            lockout_for(u32::MAX),
            Some(MAX_LOCKOUT),
            "the doubling must not overflow at an absurd attempt count"
        );
    }

    /// The ceiling is the bound on how long anyone can hold the door shut
    /// against a legitimate user. It is a policy promise, so it is pinned.
    #[test]
    fn no_lockout_ever_exceeds_a_minute() {
        for attempts in 0..2_000u32 {
            let lockout = lockout_for(attempts).unwrap_or_default();
            assert!(
                lockout <= MAX_LOCKOUT,
                "attempt {attempts} earned {lockout:?}, past the {MAX_LOCKOUT:?} ceiling"
            );
        }
        assert!(
            MAX_LOCKOUT < IDLE_RESET,
            "a counter must never reopen and reset in the same instant"
        );
    }

    #[test]
    fn the_lockout_expires() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, source(), ALICE, FREE_ATTEMPTS, now);

        let still_locked = now + BASE_LOCKOUT - Duration::from_millis(1);
        assert!(refusal(&throttle, source(), ALICE, still_locked).is_some());

        let expired = now + BASE_LOCKOUT;
        assert!(
            refusal(&throttle, source(), ALICE, expired).is_none(),
            "the door reopens once the lockout has run out"
        );
    }

    #[test]
    fn a_success_clears_both_counters() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, source(), ALICE, FREE_ATTEMPTS, now);
        assert!(refusal(&throttle, source(), ALICE, now).is_some());

        throttle.record_success(source(), ALICE);
        assert_eq!(
            allow(&throttle, source(), ALICE, now),
            None,
            "the counters restarted, so the next attempt is inside the budget again"
        );
    }

    /// One source walking a username list is caught by the source counter even
    /// though no single username reaches the budget.
    #[test]
    fn the_source_counter_adds_up_across_usernames() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        for index in 0..FREE_ATTEMPTS {
            throttle.begin_attempt(source(), &format!("user-{index}"), now);
        }
        assert!(
            refusal(&throttle, source(), "someone-new", now).is_some(),
            "the source must be locked out however many usernames it tried"
        );
    }

    /// Many sources converging on one account are caught by the username
    /// counter, even though no single source reaches the budget.
    #[test]
    fn the_username_counter_adds_up_across_sources() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        for index in 0..FREE_ATTEMPTS {
            let address = IpAddr::from([203, 0, 113, index as u8]);
            throttle.begin_attempt(Some(address), ALICE, now);
        }
        assert!(
            refusal(
                &throttle,
                Some(IpAddr::from([203, 0, 113, 200])),
                ALICE,
                now
            )
            .is_some(),
            "the account must be protected however many addresses tried it"
        );
    }

    /// A routed /64 is one subscriber. Counted by the full address, an attacker
    /// gets a fresh counter for every attempt and the per-source half of the
    /// throttle does nothing.
    #[test]
    fn an_ipv6_subscriber_prefix_is_one_source() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        let in_prefix = |host: u16| {
            Some(IpAddr::from(std::net::Ipv6Addr::new(
                0x2001, 0xdb8, 0, 0, 0, 0, 0, host,
            )))
        };
        for host in 1..=(FREE_ATTEMPTS as u16) {
            throttle.begin_attempt(in_prefix(host), ALICE, now);
        }
        assert!(
            refusal(&throttle, in_prefix(9999), "someone-new", now).is_some(),
            "rotating the host part of one /64 must not buy more attempts"
        );

        // A different /64 is a different subscriber and keeps its own budget.
        let other_prefix = Some(IpAddr::from(std::net::Ipv6Addr::new(
            0x2001, 0xdb8, 0, 1, 0, 0, 0, 1,
        )));
        assert!(refusal(&throttle, other_prefix, "someone-else", now).is_none());
    }

    /// A dual-stack listener reports IPv4 peers as IPv4-mapped IPv6 addresses,
    /// which all share the `::` /64. Counted as IPv6 they would be one bucket,
    /// so five wrong passwords from anywhere would lock out every IPv4 caller.
    #[test]
    fn ipv4_mapped_callers_are_counted_apart() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        let mapped = |last: u8| {
            Some(IpAddr::from(
                std::net::Ipv4Addr::new(198, 51, 100, last).to_ipv6_mapped(),
            ))
        };
        for _ in 0..FREE_ATTEMPTS {
            throttle.begin_attempt(mapped(1), ALICE, now);
        }
        assert!(
            refusal(&throttle, mapped(1), "someone-new", now).is_some(),
            "the caller that spent the budget must be locked out"
        );
        assert!(
            refusal(&throttle, mapped(2), "someone-else", now).is_none(),
            "a different IPv4 caller must keep its own budget"
        );
        // The mapped form and the plain form are the same caller.
        assert!(
            refusal(
                &throttle,
                Some(IpAddr::from([198, 51, 100, 1])),
                "another-name",
                now
            )
            .is_some(),
            "the mapped and unmapped forms of one address must share a counter"
        );
    }

    /// One caller's failures must not lock a different caller out of a
    /// different account.
    #[test]
    fn an_unrelated_caller_is_not_locked_out() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, source(), ALICE, FREE_ATTEMPTS + 4, now);
        assert!(refusal(&throttle, other_source(), "bob", now).is_none());
    }

    /// With no source address the username counter still applies, so a server
    /// that cannot report the peer address is throttled rather than unlimited.
    #[test]
    fn a_missing_source_address_still_counts_the_username() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, None, ALICE, FREE_ATTEMPTS, now);
        assert!(refusal(&throttle, None, ALICE, now).is_some());
    }

    /// The missing-source notice is worth giving once and no more: it describes
    /// how the server was started, not what this request did.
    #[test]
    fn the_missing_source_notice_is_given_once() {
        let throttle = LoginThrottle::default();
        assert!(throttle.note_missing_source());
        assert!(!throttle.note_missing_source());
        assert!(!throttle.note_missing_source());
    }

    /// A quiet window forgives the counter, so a mistyped password months ago
    /// does not add to today's.
    #[test]
    fn a_quiet_counter_starts_again() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, source(), ALICE, FREE_ATTEMPTS - 1, now);

        let later = now + IDLE_RESET;
        assert_eq!(
            allow(&throttle, source(), ALICE, later),
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
            throttle.begin_attempt(None, &format!("user-{index}"), now);
        }
        let tracked = throttle.counters().len();
        assert!(
            tracked <= MAX_TRACKED,
            "the throttle must not become its own memory-exhaustion path, tracked {tracked}"
        );
    }

    /// Eviction must not be a way to end a lockout early. The quiet sweep keeps
    /// a counter that is still refusing, however long ago its last attempt was.
    #[test]
    fn a_live_lockout_survives_the_quiet_sweep() {
        let throttle = LoginThrottle::default();
        let now = Instant::now();
        spend(&throttle, source(), ALICE, FREE_ATTEMPTS, now);
        assert!(refusal(&throttle, source(), ALICE, now).is_some());

        // Another caller's attempt runs the sweep. Wind the clock past the
        // quiet window so the sweep would drop the locked counter if it could.
        let much_later = now + IDLE_RESET + Duration::from_secs(1);
        let mut counters = throttle.counters();
        counters.insert(
            Counter::Username("locked-far-ahead".to_string()),
            Attempts {
                attempts: FREE_ATTEMPTS,
                last_attempt: now,
                locked_until: Some(much_later + Duration::from_secs(30)),
            },
        );
        make_room(&mut counters, much_later);
        assert!(
            counters.contains_key(&Counter::Username("locked-far-ahead".to_string())),
            "a counter that is still refusing must survive the sweep"
        );
        drop(counters);
    }

    /// A long username cannot be used to grow the map entry by entry, and the
    /// same limit is what a log line prints.
    #[test]
    fn a_username_key_is_truncated() {
        let long = "x".repeat(MAX_KEY_CHARS * 4);
        assert_eq!(short_name(&long).chars().count(), MAX_KEY_CHARS);
        match keys(None, &long).as_slice() {
            [Counter::Username(name)] => assert_eq!(name.chars().count(), MAX_KEY_CHARS),
            other => panic!("expected one username counter, got {other:?}"),
        }
    }
}
