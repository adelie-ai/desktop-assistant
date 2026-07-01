//! Common context-window / input-budget sizes and snap-down logic (issue #425).
//!
//! A learned input-token ceiling derived from an overflow error is a fuzzy,
//! provider-scraped number. Rather than pin the budget to that exact value —
//! which jitters turn to turn and is only as trustworthy as the parse — we snap
//! it DOWN to the nearest rung on a fixed ladder of round sizes. The result is a
//! stable, recognizable budget that's comfortably under the real window, and a
//! slightly-wrong extraction lands on the same rung as the right one, so the
//! fuzziness stops mattering.
//!
//! The rungs are budget buckets, not exact model windows: a model advertised as
//! "200k" might really accept 202752 total tokens, and after reserving output we
//! want an *input* budget safely below that (see
//! [`crate::error_classify::derive_input_ceiling`]). Snapping DOWN (never to the
//! nearest) guarantees we never round up past the real ceiling; the only cost is
//! a little wasted headroom, which the success high-water floor keeps bounded.

/// Ladder of budget rungs, strictly ascending. Denser through the 100k–256k
/// band where modern frontier windows cluster, so snapping there wastes little.
pub const COMMON_WINDOW_SIZES: &[u64] = &[
    4_096, 8_192, 16_384, 32_768, 65_536, 100_000, 128_000, 160_000, 192_000, 200_000, 224_000,
    256_000, 384_000, 512_000, 768_000, 1_000_000,
];

/// Snap `n` DOWN to the largest common rung `<= n`.
///
/// Returns `None` when `n` is below the smallest rung — a ceiling that small is
/// almost certainly a bad parse (not a real usable window), so the caller
/// declines to apply it as a cap rather than bricking the budget.
pub fn snap_down_to_common(n: u64) -> Option<u64> {
    COMMON_WINDOW_SIZES
        .iter()
        .rev()
        .copied()
        .find(|&rung| rung <= n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_incident_ceiling_to_192k() {
        // Issue #425: 202752 window − 8192 output reservation = 194560, which
        // must snap to 192000 (192000 + 8192 = 200192 < 202752, so the very
        // next turn fits instead of re-overflowing).
        assert_eq!(snap_down_to_common(194_560), Some(192_000));
    }

    #[test]
    fn snaps_down_never_up() {
        assert_eq!(snap_down_to_common(200_000), Some(200_000)); // exact rung
        assert_eq!(snap_down_to_common(199_999), Some(192_000)); // just under 200k
        assert_eq!(snap_down_to_common(1_500_000), Some(1_000_000)); // above top rung
        assert_eq!(snap_down_to_common(8_192), Some(8_192));
    }

    #[test]
    fn rejects_pathologically_small_values() {
        // The 534-token poison and anything below the smallest rung yield None,
        // so a garbage parse can never pin the budget.
        assert_eq!(snap_down_to_common(534), None);
        assert_eq!(snap_down_to_common(0), None);
        assert_eq!(snap_down_to_common(4_095), None);
    }

    #[test]
    fn ladder_is_strictly_ascending() {
        assert!(COMMON_WINDOW_SIZES.windows(2).all(|w| w[0] < w[1]));
    }
}
