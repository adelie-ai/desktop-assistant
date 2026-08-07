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
//! - **`reinforcement`** is what the use log knows, put on that same
//!   dimensionless scale: [`ActivationWeights::use_lift`] deviations for every
//!   e-fold of accumulated use, measured against the sum one use a day old
//!   produces. The ratio is what makes it dimensionless - a base-level sum
//!   scales with whatever unit its ages are counted in, and dividing by a
//!   reference sum cancels that. See [`ActivationWeights::reinforcement`].
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
//!   a linear count compounds. Doubling the accumulated sum adds at most
//!   `use_lift * ln 2` however large that sum already is, which holds the whole
//!   term to about three deviations over any history a store produces - see
//!   [`DEFAULT_USE_LIFT`] for what that buys and what it does not.
//!
//!   **The sum, not the number of events.** One more use raises the sum by more
//!   than a factor of two when it is far more recent than everything before it -
//!   a second open thirty seconds after the first roughly doubles a
//!   one-open history and then some. That is the recency half of the signal
//!   doing its job, and no compression of the frequency half can or should stop
//!   it. What the logarithm rules out is the other loop: an entry that keeps
//!   being retrieved because it was retrieved.
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

/// The age of the one use that anchors the reinforcement scale, in seconds.
///
/// **A base-level sum has no natural size.** Every term is an age raised to
/// `-d`, so the whole sum scales by `unit^d` when the ages are counted in a
/// different unit - stated in seconds it is a hundredth of what the same history
/// states in days. Adding that straight to a deviation count would let an
/// arbitrary choice of unit decide how much the use log is worth, which is the
/// error the semantic half exists to avoid. So the sum is read as a ratio
/// against the sum one use of this age produces, and the unit cancels.
///
/// One day is the anchor because it is the scale a knowledge base is used on:
/// an entry opened yesterday is the ordinary "recently useful" case, and it is
/// worth [`ActivationWeights::use_lift`] times `ln 2` here. Anything opened
/// within the hour scores above it and anything untouched for a month scores
/// below.
pub const USE_REFERENCE_AGE_SECONDS: f64 = 24.0 * 60.0 * 60.0;

/// How many of a source's own median absolute deviations one e-fold of
/// accumulated use is worth, relative to [`USE_REFERENCE_AGE_SECONDS`].
///
/// A deliberate scale rather than a fitted one, and what it is chosen against is
/// the semantic term's own spread. Adjacent candidates over a real store sit a
/// few tenths of a deviation apart, and a candidate the bar admits sits several
/// deviations above one it refuses. At this lift one use a day old is worth
/// about a third of a deviation - enough to settle a near-tie - and an extreme
/// history reaches about three.
///
/// **Three deviations is the guarantee, and it is worth stating exactly, because
/// the loose version of it is false.** The term has no absolute ceiling - the
/// accumulated sum has none - but it has a logarithm, so reaching past three
/// takes a history no store produces. On the measured corpus the weakest
/// candidate the bar admits sits about 4.6 deviations below the strongest, so
/// what the ceiling guarantees is that use cannot take the top line from the
/// best semantic match.
///
/// It emphatically does **not** guarantee that a used entry stays where its
/// distance put it. Ten opens inside the last half hour are worth about two and
/// a half deviations, which carries an entry sitting on the bar past most of a
/// full block. That is the design working: an entry the assistant has been
/// reading all morning should come up on a prompt that brushes it. #698's
/// caution is about the top line, and the top line is what the ceiling protects.
/// #698's log is what lets a deployment replace this figure with a measurement.
pub const DEFAULT_USE_LIFT: f64 = 0.5;

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

impl ActivationWeights {
    /// The base-level sum one use at [`USE_REFERENCE_AGE_SECONDS`] produces.
    ///
    /// The denominator [`Self::reinforcement`] divides by, so that what the use
    /// log is worth is stated against a use anybody can picture rather than
    /// against the unit the ages happen to be counted in. It follows the decay
    /// exponent, because the sum does.
    pub fn reference_sum(&self) -> f64 {
        USE_REFERENCE_AGE_SECONDS.powf(-self.use_score.safe_decay())
    }

    /// What an accumulated use sum is worth, in the source's own deviations.
    ///
    /// `use_lift * ln(1 + |sum| / reference)`, carrying the sum's sign. Four
    /// things the shape has to answer, and it answers each by construction
    /// rather than by a clamp:
    ///
    /// - **An entry nothing has used contributes nothing.** The sum is zero and
    ///   so is the answer, so a cold store ranks exactly as it did before this
    ///   score existed. No floor constant decides how far below a used entry an
    ///   unused one sits.
    /// - **One use at the reference age is worth `use_lift * ln 2`.** That is
    ///   the anchor, and it is what makes the answer independent of the unit the
    ///   ages are counted in.
    /// - **A negative mark subtracts.** "Offered, opened, and it was wrong"
    ///   drives the sum below zero, so a refuted entry ends below one nobody has
    ///   ever opened, rather than merely level with it. It does not subtract as
    ///   much as the same mark would add, and that is the writer's doing rather
    ///   than this function's: a mark is recorded as a use whichever way it
    ///   points, because the entry really was retrieved, so a negative mark of
    ///   weight `w` nets `w - 1` against a positive one's `w + 1`. The function
    ///   is symmetric about zero; the input it is given is not.
    /// - **Doubling the sum adds at most `use_lift * ln 2`.** That is the cap
    ///   #698 asks for, as a property of the function rather than a clamp. It
    ///   is a statement about the accumulated sum and not about the number of
    ///   uses: one more use raises the sum by over a factor of two when it is
    ///   far more recent than everything before it, and that is recency, which
    ///   is a signal this score is meant to carry.
    pub fn reinforcement(&self, sum: f64) -> f64 {
        let reference = self.reference_sum();
        if !sum.is_finite() || !reference.is_finite() || reference <= 0.0 {
            return 0.0;
        }
        self.use_lift * (sum.abs() / reference).ln_1p() * sum.signum()
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
    semantic + weights.reinforcement(sum)
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
    ///
    /// The window is sorted youngest-first and cut to the newest, whatever order
    /// `ages` arrives in, because that is what the writer stores: it prepends
    /// `NOW()` and cuts the tail.
    fn used(now: DateTime<Utc>, ages: &[i64], opens: u64) -> KnowledgeUseRecord {
        let mut newest_first = ages.to_vec();
        newest_first.sort_unstable();
        newest_first.truncate(RECENT_USE_WINDOW);
        KnowledgeUseRecord {
            entry_id: "kb-1".to_string(),
            offered_count: opens,
            opened_count: opens,
            marked_count: 0,
            first_seen_at: at(now, ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(at(now, 1)),
            recent_uses: newest_first.iter().map(|a| at(now, *a)).collect(),
            marks: Vec::new(),
        }
    }

    /// `count` uses laid down evenly over the last `span` seconds.
    fn evenly_over(now: DateTime<Utc>, count: u64, span: i64) -> KnowledgeUseRecord {
        let step = span / count as i64;
        let ages: Vec<i64> = (1..=count as i64).map(|i| i * step).collect();
        used(now, &ages, count)
    }

    /// `count` uses at a fixed one-hour spacing, the newest an hour ago: the
    /// same rhythm whatever the count, so a longer history is more use rather
    /// than denser use.
    fn every_hour(now: DateTime<Utc>, count: u64) -> KnowledgeUseRecord {
        let ages: Vec<i64> = (1..=count as i64).map(|i| i * 3_600).collect();
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

    /// Acceptance (#1123): doubling an entry's accumulated use buys a bounded
    /// amount, which is what stops the retrieve-mark-retrieve loop compounding.
    ///
    /// Both halves of the claim. First the function: doubling the accumulated
    /// sum adds at most `use_lift * ln 2`, at every size of sum. Then a record:
    /// an entry used twice as often over twice as long - the same rhythm, kept
    /// up for longer, which is how an entry accrues use over time - gains no
    /// more than that.
    ///
    /// See `one_much_more_recent_use_may_raise_the_score_by_more_than_that` for
    /// the case this bound deliberately does not cover.
    #[test]
    fn doubling_the_accumulated_use_raises_activation_by_a_bounded_amount() {
        let now = now();
        let weights = ActivationWeights::default();
        let bound = weights.use_lift * std::f64::consts::LN_2;
        let reference = weights.reference_sum();

        for exponent in -6..=12 {
            let sum = reference * 2.0_f64.powi(exponent);
            let step = weights.reinforcement(sum * 2.0) - weights.reinforcement(sum);
            assert!(
                step <= bound,
                "doubling a sum of {sum} added {step}, past the {bound} a doubling may buy"
            );
        }

        let mut previous = activation(0.0, Some(&every_hour(now, 1)), now, &weights);
        for count in [2u64, 4, 8, 16, 32, 64, 128, 256] {
            let doubled = activation(0.0, Some(&every_hour(now, count)), now, &weights);
            let step = doubled - previous;
            assert!(
                step <= bound,
                "going to {count} uses added {step}, past the {bound} a doubling may buy"
            );
            previous = doubled;
        }
    }

    /// The bound above is about the accumulated sum, and this is the case it
    /// does not cover: adding one use far more recent than everything before it
    /// more than doubles the sum, so the score may rise by more than
    /// `use_lift * ln 2`.
    ///
    /// Stated as a test rather than left as a gap, because a later reader
    /// meeting it in production would otherwise take it for the bound leaking.
    /// It is recency, which is half of what a per-use age buys, and no
    /// compression of the frequency half should suppress it. What bounds this
    /// case is the ceiling, not the step - see
    /// `an_extreme_use_history_stays_inside_the_stated_ceiling`.
    #[test]
    fn one_much_more_recent_use_may_raise_the_score_by_more_than_that() {
        let now = now();
        let weights = ActivationWeights::default();
        let bound = weights.use_lift * std::f64::consts::LN_2;

        let once = activation(0.0, Some(&used(now, &[60], 1)), now, &weights);
        let twice = activation(0.0, Some(&used(now, &[30, 60], 2)), now, &weights);

        assert!(
            twice - once > bound,
            "a second open thirty seconds after the first added {}, and recency is supposed to \
             be able to add more than the {bound} a doubled sum buys",
            twice - once
        );
    }

    /// The bar every candidate has already cleared, and so the score the
    /// weakest one in any block carries. `crate::recall::RECALL_BAR` states it;
    /// it is repeated here because this module must not depend on the block.
    const BAR: f64 = 6.8;

    /// What the measured prompts reached, nearest candidate first
    /// (`crate::recall`'s seeded corpus). The weakest real hit reached 7.3 and
    /// the strongest 11.4, so a best match leads the bar by anywhere from half a
    /// deviation to 4.6 - which is the range any claim about the top line has to
    /// hold over.
    const MEASURED_HITS: &[f64] = &[
        11.4, 10.9, 10.2, 9.8, 9.4, 9.1, 8.7, 8.4, 8.0, 7.8, 7.5, 7.4, 7.3,
    ];

    /// An entry opened and marked for years: the largest history a store
    /// realistically holds.
    fn a_veteran(now: DateTime<Utc>) -> KnowledgeUseRecord {
        let mut veteran = evenly_over(now, 500, YEAR);
        veteran.marked_count = 20;
        veteran.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Positive,
            reason: None,
            marked_at: at(now, 60),
        }];
        veteran
    }

    /// Acceptance (#1123), the reason the ceiling matters: an entry that has
    /// been opened and marked for years, sitting exactly on the bar, must still
    /// rank below an entry nobody has ever opened that the prompt matched
    /// better. That is #698's caution, and it is about the top line.
    ///
    /// Over every distance the measured prompts actually reached, not only the
    /// strongest of them: a claim tested at one favourable point is not a
    /// claim.
    #[test]
    fn a_heavily_used_entry_never_outranks_the_best_semantic_match() {
        let now = now();
        let weights = ActivationWeights::default();
        let veteran = a_veteran(now);

        let at_the_bar = activation(BAR, Some(&veteran), now, &weights);
        for best in MEASURED_HITS {
            let cold = activation(*best, None, now, &weights);
            assert!(
                at_the_bar < cold,
                "a veteran at the bar scored {at_the_bar} against a cold best match at \
                 {best} deviations, which scored {cold}"
            );
        }
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

    /// A negative mark drives the entry below one nobody has ever opened, rather
    /// than merely failing to lift it.
    #[test]
    fn a_refuted_entry_ranks_below_one_nobody_has_opened() {
        let now = now();
        let weights = ActivationWeights::default();

        // As the writer stores it: a mark is a use whichever way it points, so
        // the stamp is in the window and `marked_count` moved too.
        let mut refuted = used(now, &[60, 3_600], 1);
        refuted.marked_count = 1;
        refuted.marks = vec![KnowledgeMark {
            source: MarkSource::Model,
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
        let weights = ActivationWeights::default();
        let reference = weights.reference_sum();

        assert_eq!(weights.reinforcement(0.0), 0.0);
        assert!(weights.reinforcement(reference) > 0.0);
        assert!(weights.reinforcement(-reference) < 0.0);
        assert_eq!(
            weights.reinforcement(reference),
            -weights.reinforcement(-reference)
        );
        // A sum that is not a number contributes nothing rather than poisoning
        // every comparison the ranking makes.
        assert_eq!(weights.reinforcement(f64::NAN), 0.0);
        assert_eq!(weights.reinforcement(f64::INFINITY), 0.0);
    }

    /// The anchor, pinned: one use at the reference age is worth exactly
    /// `use_lift * ln 2`.
    ///
    /// This is what makes the term dimensionless. A base-level sum scales with
    /// whatever unit its ages are counted in - the same history is a hundred
    /// times larger stated in days than in seconds - so adding one straight to a
    /// deviation count would let that choice decide how much the use log is
    /// worth. Dividing by the sum of one reference use cancels it.
    #[test]
    fn one_use_at_the_reference_age_is_worth_a_stated_amount() {
        let now = now();
        let weights = ActivationWeights::default();
        let a_day = USE_REFERENCE_AGE_SECONDS as i64;

        let scored = activation(0.0, Some(&used(now, &[a_day], 1)), now, &weights);

        assert!(
            (scored - weights.use_lift * std::f64::consts::LN_2).abs() < 1e-9,
            "one use a day old scored {scored}"
        );
    }

    /// The lift has to be large enough to do something. A day-old use must
    /// settle a near-tie - two candidates a few hundredths of a deviation apart,
    /// which is what adjacent rows of a real store look like.
    #[test]
    fn a_recent_use_settles_a_near_tie_between_two_candidates() {
        let now = now();
        let weights = ActivationWeights::default();
        let a_day = USE_REFERENCE_AGE_SECONDS as i64;

        let nearer_but_unread = activation(9.10, None, now, &weights);
        let further_but_used = activation(9.00, Some(&used(now, &[a_day], 1)), now, &weights);

        assert!(
            further_but_used > nearer_but_unread,
            "a day-old use scored {further_but_used} against an unread candidate a tenth of a \
             deviation nearer at {nearer_but_unread}; a lift this small settles nothing"
        );
    }

    /// The ceiling the scale is chosen against, pinned so the claim cannot rot.
    ///
    /// The most extreme history a store realistically holds - hundreds of opens
    /// across a year, and a person's mark set a minute ago, which is the largest
    /// single term the log can carry - must stay inside the three deviations
    /// [`DEFAULT_USE_LIFT`] promises. The bar's own corpus puts the weakest
    /// admitted candidate and the strongest about 4.6 apart, so a term under
    /// three cannot carry an entry from the bottom of a block to the top.
    #[test]
    fn an_extreme_use_history_stays_inside_the_stated_ceiling() {
        let now = now();
        let weights = ActivationWeights::default();

        let mut extreme = evenly_over(now, 500, YEAR);
        extreme.marked_count = 20;
        extreme.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Positive,
            reason: None,
            marked_at: at(now, 60),
        }];

        let lift = activation(0.0, Some(&extreme), now, &weights);
        assert!(
            (0.0..3.0).contains(&lift),
            "an extreme history lifted {lift} deviations, past the three the scale is chosen \
             against"
        );
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
