//! The use log (#698): what a knowledge entry was offered for, what was
//! opened, and what was marked.
//!
//! Retrieval ranks entries with weights, and every weight has to come from
//! somewhere. A value fitted against one store, one embedding model and one
//! subject domain does not carry to a second deployment, and it does not stay
//! still inside one deployment either, because a growing store pulls every
//! nearest neighbour closer. Nothing recorded that an entry was put in front of
//! the model, and nothing recorded that the model took it up, so every
//! coefficient in the retrieval path was chosen by hand and stayed chosen by
//! hand. This module holds the record that lets a deployment measure its own.
//!
//! ## Three records, none of them inferred
//!
//! | record | what happened | quality |
//! | --- | --- | --- |
//! | offered | the entry appeared in a `[Recall]` block or a search result | low |
//! | opened | something fetched the entry by id after it was offered | high |
//! | marked | somebody said the entry was useful, or was wrong | highest |
//!
//! Usefulness is never read out of whether a retrieved fact appears to have
//! shaped an answer. `[Recall]` offers ids and no bodies precisely so that a
//! fetch is a deliberate act: the model read a one-line stand-in, and chose to
//! open the entry. That choice is a fact about what happened, not a guess.
//!
//! **The ratio is the interesting number.** An offer is mainly a denominator.
//! An entry offered many times and never opened is ranking too high and earning
//! nothing. An entry offered twice and opened twice is carrying its weight.
//!
//! ## Bounded per entry
//!
//! A spacing term needs per-use timestamps, because when the uses fell is the
//! half of the signal a lifetime counter cannot express. Keeping every event
//! for ever is unbounded, so a record keeps two things: the most recent
//! [`RECENT_USE_WINDOW`] use timestamps exactly, and aggregate counters with a
//! first-seen stamp for everything older. That is the standard hybrid for
//! ACT-R base-level activation - exact over the recent window, and the
//! streaming approximation over the tail. It is bounded per entry and it loses
//! nothing the score reads.
//!
//! ## One figure, read two ways
//!
//! [`KnowledgeUseRecord::use_sum`] states what the log knows, on one scale.
//! [`KnowledgeUseRecord::usefulness`] is its logarithm, floored, which is the
//! figure to report or to compare between entries; retrieval reads the sum
//! itself, because it joins it with a term of its own before either is
//! compressed - see [`crate::domain::activation`], which is what ranks the
//! `[Recall]` block.
//!
//! The weights the sum takes are [`UseScoreWeights`], and their defaults are
//! declared starting points rather than measured values - which is the whole
//! point of the log, because a deployment that keeps one can measure its own.

use chrono::{DateTime, Utc};

/// How many use timestamps a record keeps exactly.
///
/// Why a window rather than every event: the spacing between recent uses is
/// what a lifetime counter cannot express, and it is also the part that decays
/// out of relevance first. Ten covers the recent history any decay term still
/// weighs, and everything older is carried by the counters and the first-seen
/// stamp - see [`KnowledgeUseRecord::usefulness`].
pub const RECENT_USE_WINDOW: usize = 10;

/// The smallest sum [`KnowledgeUseRecord::usefulness`] takes a logarithm of.
///
/// The sum is zero for an entry nothing has used, and it goes negative for an
/// entry whose only record is a fresh negative mark. Both are real states and
/// both must produce a number, so the sum is floored here rather than the
/// result being made optional. The floor puts them at about -13.8, well under
/// any entry with a single use.
pub const MIN_ACTIVATION_SUM: f64 = 1e-6;

/// How much of a mark's reason is stored.
///
/// The reason comes from a language model and nothing before storage bounds it.
/// It is one statement of why an entry helped or was wrong, which is what a
/// later reader needs and all a later reader will read - the same length a
/// knowledge entry's one-line summary gets, chosen for the same reason and
/// stated here rather than borrowed, because a change to the summary's cap is a
/// summary decision.
pub const MARK_REASON_MAX_CHARS: usize = 200;

/// Whether a mark says the entry helped or says it was wrong.
///
/// A negative mark is not the absence of a positive one. "Offered, opened, and
/// it was wrong" is the strongest evidence for retiring an entry that this log
/// can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkPolarity {
    /// The entry helped.
    Positive,
    /// The entry was wrong, stale, or misleading.
    Negative,
}

impl MarkPolarity {
    /// The value stored in the database and reported on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
        }
    }

    /// The domain value behind a stored string. `None` for anything the schema
    /// does not allow, so a row written by a build that knew a value this one
    /// does not is dropped rather than raised.
    pub fn from_stored(value: &str) -> Option<Self> {
        match value {
            "positive" => Some(Self::Positive),
            "negative" => Some(Self::Negative),
            _ => None,
        }
    }

    /// `+1` for a positive mark and `-1` for a negative one, so a score can
    /// add both terms the same way.
    pub fn sign(self) -> f64 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }
}

/// Who made a mark.
///
/// A person's mark outranks the model's, and the two are kept apart rather
/// than averaged: the model marks what it believes helped this turn, and a
/// person marks what they know about the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkSource {
    /// The assistant marked the entry from its own turn.
    Model,
    /// A person marked the entry.
    ///
    /// No client offers this yet. The value exists so the record has somewhere
    /// to put a human judgement the day one does, rather than a schema change
    /// standing between the two.
    Person,
}

impl MarkSource {
    /// The value stored in the database and reported on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Person => "person",
        }
    }

    /// The domain value behind a stored string. `None` for anything the schema
    /// does not allow, so a row written by a build that knew a value this one
    /// does not is dropped rather than raised.
    pub fn from_stored(value: &str) -> Option<Self> {
        match value {
            "model" => Some(Self::Model),
            "person" => Some(Self::Person),
            _ => None,
        }
    }
}

/// One standing judgement about an entry, from one source.
///
/// A source holds one mark at a time. A second mark from the same source
/// replaces the first, because a judgement is a current opinion rather than an
/// event: an entry marked wrong last month and right today is right today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeMark {
    /// Who made it.
    pub source: MarkSource,
    /// Whether it says the entry helped or says it was wrong.
    pub polarity: MarkPolarity,
    /// Why, in the marker's own words, when one was given. A negative mark's
    /// reason is the capture point negative memory reads.
    pub reason: Option<String>,
    /// When the standing mark was last set.
    pub marked_at: DateTime<Utc>,
}

/// What the log knows about one entry.
///
/// A record exists once an entry has been offered, opened or marked at least
/// once. An entry with no record has never been in front of anybody, which is
/// a different state from an entry that was offered and ignored - and the two
/// must not be confused, because only the second is evidence.
#[derive(Debug, Clone)]
pub struct KnowledgeUseRecord {
    /// The knowledge entry this record is about.
    pub entry_id: String,
    /// How many times the entry has appeared in a `[Recall]` block or a search
    /// result. The denominator of the take-up rate.
    pub offered_count: u64,
    /// How many times something fetched the entry by id after it was offered.
    pub opened_count: u64,
    /// How many times a mark was set on the entry, of either polarity.
    pub marked_count: u64,
    /// When the entry first entered the log.
    pub first_seen_at: DateTime<Utc>,
    /// When the entry was last offered, or `None` when it never has been.
    pub last_offered_at: Option<DateTime<Utc>>,
    /// The most recent use timestamps, newest first, at most
    /// [`RECENT_USE_WINDOW`] of them.
    ///
    /// A "use" is an open or a mark. An offer is not a use: being shown is not
    /// being taken up, and counting it as one would let ranking feed itself.
    pub recent_uses: Vec<DateTime<Utc>>,
    /// The standing marks, at most one per source.
    pub marks: Vec<KnowledgeMark>,
}

/// The coefficients [`KnowledgeUseRecord::usefulness`] applies.
///
/// Every one of them is a declared starting point, not a measured value. They
/// are a struct rather than constants so that a deployment which has kept a use
/// log can fit its own and pass them in, which is what the log exists for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UseScoreWeights {
    /// The power-law decay exponent, `d` in `t^-d`. Must sit strictly between
    /// 0 and 1: the tail approximation divides by `1 - d`, and a `d` of 0 makes
    /// the score a plain count with no recency in it at all.
    ///
    /// 0.5 is the ACT-R standard value.
    pub decay: f64,
    /// How many uses a mark by the model is worth.
    pub model_mark: f64,
    /// How many uses a mark by a person is worth. Higher than
    /// [`Self::model_mark`], because a person's mark outranks the rest.
    pub person_mark: f64,
}

impl Default for UseScoreWeights {
    fn default() -> Self {
        Self {
            decay: 0.5,
            model_mark: 2.0,
            person_mark: 8.0,
        }
    }
}

impl UseScoreWeights {
    /// The decay exponent, held inside the range the formula is defined over.
    ///
    /// Public because anything reading [`KnowledgeUseRecord::use_sum`] has to
    /// use the same exponent the sum was computed with - see
    /// [`crate::domain::activation::ActivationWeights::reference_sum`], which
    /// states a reference sum at this exponent.
    pub fn safe_decay(&self) -> f64 {
        self.decay.clamp(0.01, 0.99)
    }

    /// What one mark from `source` is worth, in uses.
    fn mark_weight(&self, source: MarkSource) -> f64 {
        match source {
            MarkSource::Model => self.model_mark,
            MarkSource::Person => self.person_mark,
        }
    }
}

/// Seconds between `then` and `now`, never less than one.
///
/// One second is the floor because the score raises this to a negative power,
/// so an age of zero is an infinity and a negative age - which a clock that
/// stepped backwards produces - is not a number.
fn age_seconds(now: DateTime<Utc>, then: DateTime<Utc>) -> f64 {
    let seconds = (now - then).num_milliseconds() as f64 / 1000.0;
    seconds.max(1.0)
}

impl KnowledgeUseRecord {
    /// An entry the log has never seen, as of `now`.
    ///
    /// Every counter is zero and there are no marks, so
    /// [`Self::usefulness`] answers the floor. Callers that read a batch of
    /// records use this for the ids that came back with nothing, so a missing
    /// row and a row of zeroes score alike.
    pub fn unseen(entry_id: impl Into<String>, now: DateTime<Utc>) -> Self {
        Self {
            entry_id: entry_id.into(),
            offered_count: 0,
            opened_count: 0,
            marked_count: 0,
            first_seen_at: now,
            last_offered_at: None,
            recent_uses: Vec::new(),
            marks: Vec::new(),
        }
    }

    /// Opens plus marks: every time the entry was taken up rather than merely
    /// shown.
    pub fn total_uses(&self) -> u64 {
        self.opened_count.saturating_add(self.marked_count)
    }

    /// How often an offer led to the entry being opened, or `None` when the
    /// entry has never been offered and the ratio has no denominator.
    ///
    /// It can exceed 1. An entry offered once and then fetched on three later
    /// turns has earned all three opens, and capping the figure would hide
    /// exactly the entries that are carrying their weight.
    pub fn take_up_rate(&self) -> Option<f64> {
        (self.offered_count > 0).then(|| self.opened_count as f64 / self.offered_count as f64)
    }

    /// The mark that speaks for this entry: a person's where there is one, and
    /// the model's otherwise.
    pub fn standing_mark(&self) -> Option<&KnowledgeMark> {
        self.marks
            .iter()
            .find(|m| m.source == MarkSource::Person)
            .or_else(|| self.marks.iter().find(|m| m.source == MarkSource::Model))
    }

    /// What the log says this entry is worth, as of `now`.
    ///
    /// The figure is an ACT-R base-level activation over the entry's uses, with
    /// the marks folded in as weighted uses that carry a sign. In full:
    ///
    /// ```text
    /// S = sum over the recent window of  age^-d
    ///   + the tail approximation over every older use
    ///   + sum over the marks of  sign * weight * age^-d
    ///
    /// score = ln(max(S, MIN_ACTIVATION_SUM))
    /// ```
    ///
    /// Three properties follow from that shape, and each is a rule this issue
    /// had to keep:
    ///
    /// - **Recency weighted.** Every term is an age raised to a negative power,
    ///   so an old use and an old mark both fade. Nothing here is a lifetime
    ///   total that would outweigh them.
    /// - **A negative mark lowers the score.** It subtracts, rather than
    ///   failing to add, so an entry that was opened and then found wrong ends
    ///   up below one that was never opened at all.
    /// - **No rich-get-richer.** The logarithm means doubling the uses adds a
    ///   constant, so an entry cannot run away with the ranking merely because
    ///   it ranked once.
    ///
    /// The tail approximation is the standard one: the uses older than the
    /// recent window are treated as evenly spread between the first-seen stamp
    /// and the oldest timestamp still held, which integrates to
    /// `(n - k) * (T^(1-d) - t_k^(1-d)) / ((1 - d) * (T - t_k))`. With an empty
    /// window it reduces to `n / (1 - d) * T^-d`, whose logarithm is the
    /// familiar `ln(n / (1 - d)) - d * ln(T)`.
    pub fn usefulness(&self, now: DateTime<Utc>, weights: &UseScoreWeights) -> f64 {
        self.use_sum(now, weights).max(MIN_ACTIVATION_SUM).ln()
    }

    /// The base-level sum itself: `S` in the formula [`Self::usefulness`]
    /// documents, before any logarithm.
    ///
    /// The sum rather than its logarithm, because the retrieval score composes
    /// it with a term of its own and the two have to be joined before either is
    /// compressed - see [`crate::domain::activation`].
    ///
    /// It is zero for an entry nothing has used and negative for one whose only
    /// record is a standing negative mark; both are real states and neither is
    /// floored here, so a caller can tell them apart.
    pub fn use_sum(&self, now: DateTime<Utc>, weights: &UseScoreWeights) -> f64 {
        let d = weights.safe_decay();

        let window: Vec<f64> = self
            .recent_uses
            .iter()
            .map(|t| age_seconds(now, *t))
            .collect();
        let recent: f64 = window.iter().map(|age| age.powf(-d)).sum();

        let tail = self.tail_term(now, d, window.iter().copied().fold(0.0, f64::max));

        let marks: f64 = self
            .marks
            .iter()
            .map(|mark| {
                let age = age_seconds(now, mark.marked_at);
                mark.polarity.sign() * weights.mark_weight(mark.source) * age.powf(-d)
            })
            .sum();

        recent + tail + marks
    }

    /// The approximated contribution of the uses that fell out of the recent
    /// window, given the age of the oldest use still in it (`oldest`, zero when
    /// the window is empty).
    fn tail_term(&self, now: DateTime<Utc>, d: f64, oldest: f64) -> f64 {
        let held = self.recent_uses.len() as u64;
        let total = self.total_uses();
        if total <= held {
            return 0.0;
        }
        let missing = (total - held) as f64;
        let lifetime = age_seconds(now, self.first_seen_at);
        // The window already covers everything back to the first use, so there
        // is no span left for the older uses to be spread over.
        if lifetime <= oldest {
            return 0.0;
        }
        if oldest <= 0.0 {
            // No exact timestamps at all: the uses are spread over the whole
            // lifetime, which is the streaming approximation on its own.
            return missing / (1.0 - d) * lifetime.powf(-d);
        }
        missing * (lifetime.powf(1.0 - d) - oldest.powf(1.0 - d))
            / ((1.0 - d) * (lifetime - oldest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    fn at(now: DateTime<Utc>, seconds_ago: i64) -> DateTime<Utc> {
        now - TimeDelta::seconds(seconds_ago)
    }

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-06T12:00:00Z")
            .expect("fixed clock parses")
            .with_timezone(&Utc)
    }

    /// A record with `opens` opens, the newest `window` of them held exactly.
    fn opened(now: DateTime<Utc>, ages: &[i64], opens: u64) -> KnowledgeUseRecord {
        KnowledgeUseRecord {
            entry_id: "kb-1".to_string(),
            offered_count: opens,
            opened_count: opens,
            marked_count: 0,
            first_seen_at: at(now, ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(at(now, 1)),
            recent_uses: ages.iter().map(|a| at(now, *a)).collect(),
            marks: Vec::new(),
        }
    }

    #[test]
    fn an_entry_nothing_has_used_scores_the_floor() {
        let now = now();
        let record = KnowledgeUseRecord::unseen("kb-1", now);
        assert_eq!(
            record.usefulness(now, &UseScoreWeights::default()),
            1e-6f64.ln()
        );
        assert_eq!(record.take_up_rate(), None);
    }

    #[test]
    fn offering_opening_and_marking_are_counted_separately() {
        let now = now();
        let record = KnowledgeUseRecord {
            offered_count: 10,
            opened_count: 2,
            marked_count: 1,
            ..opened(now, &[60], 0)
        };
        assert_eq!(record.offered_count, 10);
        assert_eq!(record.opened_count, 2);
        assert_eq!(record.marked_count, 1);
        assert_eq!(record.total_uses(), 3);
        assert_eq!(record.take_up_rate(), Some(0.2));
    }

    #[test]
    fn a_negative_mark_lowers_the_score_below_the_same_entry_unmarked() {
        let now = now();
        let plain = opened(now, &[60, 600], 2);
        let mut marked = plain.clone();
        marked.marked_count = 1;
        marked.marks = vec![KnowledgeMark {
            source: MarkSource::Model,
            polarity: MarkPolarity::Negative,
            reason: Some("named the wrong host".to_string()),
            marked_at: at(now, 60),
        }];

        let weights = UseScoreWeights::default();
        assert!(
            marked.usefulness(now, &weights) < plain.usefulness(now, &weights),
            "a negative mark must lower the score, not merely fail to raise it"
        );
    }

    #[test]
    fn a_positive_mark_raises_the_score_and_a_person_outweighs_the_model() {
        let now = now();
        let plain = opened(now, &[60, 600], 2);
        let mut by_model = plain.clone();
        by_model.marked_count = 1;
        by_model.marks = vec![KnowledgeMark {
            source: MarkSource::Model,
            polarity: MarkPolarity::Positive,
            reason: None,
            marked_at: at(now, 60),
        }];
        let mut by_person = by_model.clone();
        by_person.marks[0].source = MarkSource::Person;

        let weights = UseScoreWeights::default();
        assert!(plain.usefulness(now, &weights) < by_model.usefulness(now, &weights));
        assert!(by_model.usefulness(now, &weights) < by_person.usefulness(now, &weights));
    }

    #[test]
    fn scoring_is_recency_weighted_so_an_old_mark_decays() {
        let now = now();
        let base = opened(now, &[3600], 1);

        let mut fresh = base.clone();
        fresh.marked_count = 1;
        fresh.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Positive,
            reason: None,
            marked_at: at(now, 60),
        }];

        let mut stale = fresh.clone();
        stale.marks[0].marked_at = at(now, 365 * 24 * 3600);

        let weights = UseScoreWeights::default();
        assert!(
            stale.usefulness(now, &weights) < fresh.usefulness(now, &weights),
            "a year-old mark must weigh less than a minute-old one"
        );
        // And it decays towards, not below, the unmarked entry.
        assert!(stale.usefulness(now, &weights) > base.usefulness(now, &weights));
    }

    #[test]
    fn a_recent_use_outweighs_an_old_one() {
        let now = now();
        let recent = opened(now, &[60], 1);
        let old = opened(now, &[365 * 24 * 3600], 1);
        let weights = UseScoreWeights::default();
        assert!(old.usefulness(now, &weights) < recent.usefulness(now, &weights));
    }

    #[test]
    fn uses_beyond_the_window_still_count_through_the_tail() {
        let now = now();
        // Ten uses held exactly, and a hundred in total: the ninety that fell
        // out of the window must still lift the score.
        let held: Vec<i64> = (1..=RECENT_USE_WINDOW as i64).map(|i| i * 60).collect();
        let windowed = opened(now, &held, RECENT_USE_WINDOW as u64);
        let mut with_tail = windowed.clone();
        with_tail.opened_count = 100;
        with_tail.first_seen_at = at(now, 400 * 24 * 3600);

        let weights = UseScoreWeights::default();
        assert!(
            windowed.usefulness(now, &weights) < with_tail.usefulness(now, &weights),
            "uses older than the window must still be worth something"
        );
        assert!(
            with_tail.usefulness(now, &weights).is_finite(),
            "the tail approximation must stay a number"
        );
    }

    #[test]
    fn the_score_grows_logarithmically_so_uses_cannot_run_away() {
        let now = now();
        let weights = UseScoreWeights::default();
        let one = opened(now, &[60], 1);
        let mut ten = one.clone();
        ten.opened_count = 10;
        ten.first_seen_at = at(now, 600);
        let mut hundred = one.clone();
        hundred.opened_count = 100;
        hundred.first_seen_at = at(now, 6000);

        let first_step = ten.usefulness(now, &weights) - one.usefulness(now, &weights);
        let second_step = hundred.usefulness(now, &weights) - ten.usefulness(now, &weights);
        assert!(
            second_step < first_step + 1.0,
            "ten times the uses must not be ten times the score"
        );
    }

    #[test]
    fn a_person_mark_outranks_a_model_mark_as_the_standing_judgement() {
        let now = now();
        let mut record = opened(now, &[60], 1);
        record.marks = vec![
            KnowledgeMark {
                source: MarkSource::Model,
                polarity: MarkPolarity::Positive,
                reason: None,
                marked_at: at(now, 10),
            },
            KnowledgeMark {
                source: MarkSource::Person,
                polarity: MarkPolarity::Negative,
                reason: Some("out of date".to_string()),
                marked_at: at(now, 3600),
            },
        ];
        let standing = record.standing_mark().expect("a standing mark");
        assert_eq!(standing.source, MarkSource::Person);
        assert_eq!(standing.polarity, MarkPolarity::Negative);
    }

    #[test]
    fn a_clock_that_stepped_backwards_still_produces_a_number() {
        let now = now();
        let mut record = opened(now, &[60], 1);
        // A use stamped in the future: the age floor must hold.
        record.recent_uses = vec![now + TimeDelta::seconds(600)];
        record.first_seen_at = now + TimeDelta::seconds(600);
        assert!(
            record
                .usefulness(now, &UseScoreWeights::default())
                .is_finite()
        );
    }

    #[test]
    fn wire_values_for_a_mark_round_trip() {
        for polarity in [MarkPolarity::Positive, MarkPolarity::Negative] {
            assert_eq!(MarkPolarity::from_stored(polarity.as_str()), Some(polarity));
        }
        for source in [MarkSource::Model, MarkSource::Person] {
            assert_eq!(MarkSource::from_stored(source.as_str()), Some(source));
        }
        assert_eq!(MarkPolarity::from_stored("maybe"), None);
        assert_eq!(MarkSource::from_stored("committee"), None);
    }
}
