//! One activation score for retrieval (#1123).
//!
//! Retrieval had two scoring stories that never met. The live one ranked on
//! semantic distance and full-text rank, and admitted a candidate on one
//! dimensionless bar ([`crate::recall::RECALL_BAR`]). The measured one -
//! [`crate::domain::knowledge_use`] - turned the use log into a number that no
//! ranking path read. This module is the single function that settles both, so
//! there is one place to read, one place to argue with, and one place to fit.
//!
//! ## The score
//!
//! ```text
//! A_i = semantic + reinforcement
//! ```
//!
//! - **`semantic`** is how far the candidate stands out of its own source,
//!   counted in that source's median absolute deviations - the quantity
//!   [`RecallDispersion::deviations_below_median`] answers. It is never a raw
//!   distance. A distance depends on the embedding model, on how much text a row
//!   holds, and on how wide the store's subject matter is, so it means nothing
//!   across a source boundary; the deviation count means the same thing against
//!   any source. That is what lets a fourth or a fifth source join later without
//!   refitting every weight.
//! - **`reinforcement`** is what the use log knows, on the same dimensionless
//!   scale: [`ActivationWeights::use_lift`] deviations for every e-fold of
//!   accumulated use. See [`reinforcement`].
//!
//! [`RecallDispersion::deviations_below_median`]:
//!     crate::ports::recall::RecallDispersion::deviations_below_median
//!
//! ## Why base-level learning rather than a hand-tuned blend
//!
//! The reinforcement half is Anderson's ACT-R base-level activation, computed by
//! [`KnowledgeUseRecord::use_sum`]: every use is weighted by its own age raised
//! to a negative power, and the sum is read through a logarithm. Three
//! properties come out of that shape rather than out of a tuning pass.
//!
//! - **It is fitted, not invented.** The power law was derived from human
//!   forgetting curves, which is the closest available answer to "how likely is
//!   this to be wanted now".
//! - **Spacing comes free.** Each use carries its own age, so twenty uses spread
//!   over a year outrank twenty uses in one afternoon a year ago. A lifetime
//!   counter cannot express that, and a plain recency weight loses the frequency
//!   half.
//! - **The logarithm is the cap, rather than a clamp.** Marks raise ranking,
//!   ranking drives retrieval, and retrieval is what lets an entry be marked, so
//!   a linear count compounds. Doubling the accumulated use adds at most
//!   `use_lift * ln 2` however large the count already is, so a heavily used
//!   entry cannot swamp a strong semantic match.
//!
//! ## The terms that have no input yet
//!
//! The full form the epic describes carries three more terms. None of them has
//! anything to read today, so none of them is a parameter here - a weight with
//! no input is a number nobody can fit and everybody has to maintain.
//!
//! | term | where its input will come from |
//! | --- | --- |
//! | full-text rank | a recall lookup uses one mode at a time, so a vector candidate carries no rank and a lexical one carries no distance - see [`RecallRelevance`] |
//! | situation match | #1125 |
//! | salience | #1127 |
//! | interference penalty | the entry disposition of #893, which no column holds yet |
//!
//! Each of them adds to `A_i` when it exists. The semantic term is already
//! dimensionless, so a new term states its own weight in the same deviations and
//! nothing already fitted has to move.
//!
//! [`RecallRelevance`]: crate::ports::recall::RecallRelevance

use chrono::{DateTime, Utc};

use crate::domain::knowledge_use::{KnowledgeUseRecord, UseScoreWeights};

/// How many of a source's own median absolute deviations one e-fold of
/// accumulated use is worth.
///
/// One deviation is a deliberate scale rather than a fitted one. Adjacent
/// candidates over a real store sit a fraction of a deviation apart, and a
/// candidate that clears the bar sits several deviations above one that does
/// not, so at this lift the use log settles near-ties and never overturns a
/// clear semantic win. #698's log is what lets a deployment replace it with a
/// measurement.
pub const DEFAULT_USE_LIFT: f64 = 1.0;

/// The coefficients [`activation`] applies.
///
/// A struct rather than constants for the same reason
/// [`UseScoreWeights`] is one: a deployment that has kept a use log can fit its
/// own and pass them in, which is what the log exists for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationWeights {
    /// How many deviations one e-fold of accumulated use is worth. See
    /// [`DEFAULT_USE_LIFT`].
    pub use_lift: f64,
    /// What one use, and one mark, are worth to the base-level sum.
    pub use_score: UseScoreWeights,
}

impl Default for ActivationWeights {
    fn default() -> Self {
        Self {
            use_lift: DEFAULT_USE_LIFT,
            use_score: UseScoreWeights::default(),
        }
    }
}

/// What one candidate is worth, as of `now`.
///
/// `semantic` is the source-normalized signal: how many of its own source's
/// median absolute deviations the candidate stands below that source's median.
/// `record` is what the use log knows about it, and `None` - an entry nothing
/// has ever offered, opened or marked - contributes exactly zero, so a store
/// with no use history ranks on the semantic signal alone.
pub fn activation(
    semantic: f64,
    record: Option<&KnowledgeUseRecord>,
    now: DateTime<Utc>,
    weights: &ActivationWeights,
) -> f64 {
    let sum = record.map_or(0.0, |record| record.use_sum(now, &weights.use_score));
    semantic + reinforcement(sum, weights.use_lift)
}

/// What an accumulated use sum is worth, in the source's own deviations.
///
/// `lift * ln(1 + |sum|)`, carrying the sum's sign. Three edges the shape has to
/// answer, and it answers each by construction rather than by a clamp:
///
/// - **An entry nothing has used contributes nothing.** The sum is zero and so
///   is the answer, so a cold store ranks exactly as it did before this score
///   existed. There is no floor constant deciding how far below a used entry an
///   unused one sits.
/// - **A negative mark subtracts.** "Offered, opened, and it was wrong" drives
///   the sum below zero, and the answer follows it down on the same logarithmic
///   scale a positive mark climbs - so a refuted entry ends below one nobody has
///   ever opened, rather than merely level with it.
/// - **Doubling the use adds at most `lift * ln 2`.** That is the cap #698 asks
///   for, as a property of the function.
pub fn reinforcement(sum: f64, lift: f64) -> f64 {
    if !sum.is_finite() {
        return 0.0;
    }
    lift * sum.abs().ln_1p() * sum.signum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::knowledge_use::{
        KnowledgeMark, MarkPolarity, MarkSource, RECENT_USE_WINDOW,
    };
    use chrono::TimeDelta;

    const DAY: i64 = 24 * 3600;
    const YEAR: i64 = 365 * DAY;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-07T12:00:00Z")
            .expect("a fixed clock parses")
            .with_timezone(&Utc)
    }

    fn at(now: DateTime<Utc>, seconds_ago: i64) -> DateTime<Utc> {
        now - TimeDelta::seconds(seconds_ago)
    }

    /// A record of `opens` opens, the newest [`RECENT_USE_WINDOW`] of `ages`
    /// held exactly, first seen when the oldest of them landed.
    fn used(now: DateTime<Utc>, ages: &[i64], opens: u64) -> KnowledgeUseRecord {
        KnowledgeUseRecord {
            entry_id: "kb-1".to_string(),
            offered_count: opens,
            opened_count: opens,
            marked_count: 0,
            first_seen_at: at(now, ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(at(now, 1)),
            recent_uses: ages
                .iter()
                .take(RECENT_USE_WINDOW)
                .map(|a| at(now, *a))
                .collect(),
            marks: Vec::new(),
        }
    }

    /// `count` uses laid down evenly over the last `span` seconds.
    fn evenly_over(now: DateTime<Utc>, count: u64, span: i64) -> KnowledgeUseRecord {
        let step = span / count as i64;
        let ages: Vec<i64> = (1..=count as i64).map(|i| i * step).collect();
        used(now, &ages, count)
    }

    /// Acceptance (#1123): the base-level term is arithmetic over stored
    /// counters and stored timestamps. No model, no network, no clock but the
    /// one handed in.
    #[test]
    fn base_level_activation_is_computed_from_stored_counters_with_no_model_call() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[60, 600, 6_000], 3);

        let first = activation(7.0, Some(&record), now, &weights);
        let again = activation(7.0, Some(&record), now, &weights);

        assert_eq!(
            first, again,
            "the score is a pure function of what it is given"
        );
        assert!(first.is_finite());
        assert!(
            first > 7.0,
            "three recent opens must raise the candidate above its semantic signal alone, \
             got {first}"
        );
    }

    /// Acceptance (#1123): spacing. Both entries have been known for a year and
    /// both were used twenty times; one was used steadily across the year, the
    /// other twenty times inside a single day at the start of it.
    ///
    /// The lifetimes are held equal on purpose. Comparing a spread history
    /// against a burst that happened *today* would measure recency, which every
    /// decay rule has - the spacing effect is what a per-use age buys over a
    /// lifetime counter, and it only shows against an equal elapsed time.
    #[test]
    fn twenty_uses_spread_over_a_year_rank_above_twenty_uses_in_one_day() {
        let now = now();
        let weights = ActivationWeights::default();

        let spread = evenly_over(now, 20, YEAR);
        // Twenty uses inside one day, a year ago: the same count, the same age
        // since the first of them, all of it massed.
        let massed = {
            let ages: Vec<i64> = (0..20).map(|i| YEAR - i * (DAY / 20)).collect();
            used(now, &ages, 20)
        };

        assert_eq!(spread.total_uses(), massed.total_uses(), "precondition");
        assert_eq!(
            spread.first_seen_at, massed.first_seen_at,
            "precondition: equal elapsed time, so only the spacing differs"
        );

        let spread_score = activation(7.0, Some(&spread), now, &weights);
        let massed_score = activation(7.0, Some(&massed), now, &weights);
        assert!(
            spread_score > massed_score,
            "twenty uses spread over a year scored {spread_score}, twenty massed into a day \
             scored {massed_score}"
        );
    }

    /// Acceptance (#1123): doubling the use count buys a bounded amount, so a
    /// heavily used entry cannot swamp a strong semantic match.
    #[test]
    fn doubling_the_use_count_raises_activation_by_a_bounded_amount() {
        let now = now();
        let weights = ActivationWeights::default();
        let bound = weights.use_lift * std::f64::consts::LN_2;

        let mut previous = activation(0.0, Some(&evenly_over(now, 1, DAY)), now, &weights);
        for count in [2u64, 4, 8, 16, 32, 64, 128, 256] {
            let doubled = activation(0.0, Some(&evenly_over(now, count, DAY)), now, &weights);
            let step = doubled - previous;
            assert!(
                step <= bound,
                "going to {count} uses added {step}, past the {bound} a doubling may buy"
            );
            previous = doubled;
        }
    }

    /// Acceptance (#1123), the reason the bound matters: an entry that has been
    /// opened and marked for years, sitting exactly on the bar, must still rank
    /// below an entry nobody has ever opened that the prompt names outright.
    #[test]
    fn a_heavily_used_entry_does_not_outrank_a_strong_semantic_match() {
        let now = now();
        let weights = ActivationWeights::default();

        let mut veteran = evenly_over(now, 500, YEAR);
        veteran.marked_count = 20;
        veteran.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Positive,
            reason: None,
            marked_at: at(now, 60),
        }];

        let at_the_bar = activation(6.8, Some(&veteran), now, &weights);
        let strong_and_cold = activation(11.4, None, now, &weights);

        assert!(
            at_the_bar < strong_and_cold,
            "a veteran at the bar scored {at_the_bar} against a cold strong match at \
             {strong_and_cold}"
        );
    }

    /// Acceptance (#1123): the semantic term is the source-normalized deviation
    /// and never a raw distance. Two sources of quite different geometry, one
    /// candidate equally placed in each - the score is the same number.
    #[test]
    fn the_semantic_term_is_the_source_normalized_deviation_and_not_a_raw_distance() {
        use crate::ports::recall::RecallDispersion;

        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[600], 1);

        let tight = RecallDispersion::assumed(0.80, 0.05);
        let loose = RecallDispersion::assumed(0.42, 0.30);
        let stands_out_by = 7.5;

        // The same position in each source's own geometry, which is a different
        // raw distance in each.
        let near = tight.distance_at(stands_out_by);
        let far = loose.distance_at(stands_out_by);
        assert!(
            (near - far).abs() > 0.5,
            "precondition: the two sources put an equally exceptional row at very different \
             distances ({near} against {far})"
        );

        let in_tight = activation(
            tight.deviations_below_median(near),
            Some(&record),
            now,
            &weights,
        );
        let in_loose = activation(
            loose.deviations_below_median(far),
            Some(&record),
            now,
            &weights,
        );

        assert!(
            (in_tight - in_loose).abs() < 1e-9,
            "an equally-placed candidate scored {in_tight} against one source and {in_loose} \
             against another"
        );
    }

    /// Acceptance (#1123): an entry the log has never seen is ranked on the
    /// semantic signal alone, so a store with no history ranks exactly as it did
    /// before this score existed.
    #[test]
    fn an_entry_with_no_recorded_use_is_ranked_on_the_semantic_signal_alone() {
        let now = now();
        let weights = ActivationWeights::default();

        for semantic in [0.0, 6.8, 11.4, -3.0] {
            assert_eq!(
                activation(semantic, None, now, &weights),
                semantic,
                "an unused entry must contribute nothing of its own"
            );
        }

        // A row of zeroes is the same state as no row at all, and must score
        // the same - the two differ only in whether a write ever happened.
        let unseen = KnowledgeUseRecord::unseen("kb-1", now);
        assert_eq!(
            activation(6.8, Some(&unseen), now, &weights),
            activation(6.8, None, now, &weights)
        );
    }

    /// A cold store keeps its order. Ranking by activation must be the identity
    /// on a set where nothing has ever been used, or the block would reorder
    /// itself the day this landed.
    #[test]
    fn a_cold_store_keeps_the_order_its_distances_gave_it() {
        let now = now();
        let weights = ActivationWeights::default();
        let nearest_first = [11.4, 10.9, 10.2, 9.8, 9.4, 7.3];

        let scored: Vec<f64> = nearest_first
            .iter()
            .map(|s| activation(*s, None, now, &weights))
            .collect();
        let mut ranked = scored.clone();
        ranked.sort_by(|a, b| b.total_cmp(a));

        assert_eq!(scored, ranked, "a cold store must not be reordered");
    }

    /// A negative mark drives the entry below one nobody has ever opened, rather
    /// than merely failing to lift it.
    #[test]
    fn a_refuted_entry_ranks_below_one_nobody_has_opened() {
        let now = now();
        let weights = ActivationWeights::default();

        let mut refuted = used(now, &[3_600], 1);
        refuted.marked_count = 1;
        refuted.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Negative,
            reason: Some("the fact it states was withdrawn".to_string()),
            marked_at: at(now, 60),
        }];

        assert!(
            activation(7.0, Some(&refuted), now, &weights) < activation(7.0, None, now, &weights)
        );
    }

    /// The reinforcement term's own edges, stated where the reader meets them.
    #[test]
    fn the_reinforcement_term_is_zero_at_zero_and_signed_beyond_it() {
        assert_eq!(reinforcement(0.0, 1.0), 0.0);
        assert!(reinforcement(1.0, 1.0) > 0.0);
        assert!(reinforcement(-1.0, 1.0) < 0.0);
        assert_eq!(reinforcement(1.0, 1.0), -reinforcement(-1.0, 1.0));
        // A sum that is not a number contributes nothing rather than poisoning
        // every comparison the ranking makes.
        assert_eq!(reinforcement(f64::NAN, 1.0), 0.0);
        assert_eq!(reinforcement(f64::INFINITY, 1.0), 0.0);
    }

    /// Acceptance (#1123): scoring a corpus far larger than any one lookup reads
    /// costs a fraction of the budget the lookup already has.
    ///
    /// The recall scan is held to
    /// `RECALL_SCAN_STATEMENT_TIMEOUT` (four seconds) and reads at most
    /// `RECALL_ENTRY_SCAN_LIMIT` rows, so ten thousand is two hundred times what
    /// a real lookup scores. The bound below is generous enough that a loaded
    /// build machine does not fail it and tight enough that a score which became
    /// a database round trip would.
    #[test]
    fn scoring_ten_thousand_entries_stays_inside_the_search_latency_budget() {
        let now = now();
        let weights = ActivationWeights::default();
        let corpus: Vec<KnowledgeUseRecord> = (0..10_000)
            .map(|i| evenly_over(now, 1 + (i % 40), YEAR))
            .collect();

        let started = std::time::Instant::now();
        let total: f64 = corpus
            .iter()
            .enumerate()
            .map(|(i, record)| activation(7.0 + (i % 5) as f64, Some(record), now, &weights))
            .sum();
        let spent = started.elapsed();

        assert!(total.is_finite(), "every score must be a number");
        assert!(
            spent < std::time::Duration::from_millis(500),
            "scoring ten thousand entries spent {spent:?}, which is a real share of the four \
             seconds the scan itself has"
        );
    }
}
