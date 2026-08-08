//! What makes one stored fact worth more attention than another (#1127).
//!
//! Emotionally significant events consolidate preferentially in people. The
//! software analogues are cheap: an entry a person asked for, an entry with a
//! date on it, an entry about money, health, or a promise made to somebody
//! else. None of them needs a model call, because each is a property of text
//! and provenance the store already holds.
//!
//! This module is the whole rule. It says which signals exist, how each is
//! detected, and what a reading of them is worth on the one dimensionless scale
//! [`crate::domain::activation`] scores in. Two consumers read it: the `[Recall]`
//! block, where salience is the fourth term of `A_i`, and the daily
//! consolidation pass, where it is one of three terms in
//! [`crate::domain::replay`].
//!
//! ## Read, never written
//!
//! A reading is taken from the entry itself at the moment it is scored. Nothing
//! is stored, no column is added, and no write path consults it. Three things
//! follow, and the third is the one the issue asks for:
//!
//! - **A detector improves the whole store at once.** A signal added later
//!   applies to every entry ever written, including the ones written before the
//!   detector existed. A stored flag would apply only to entries written after
//!   it, and the older half of a store would read as unsalient for ever.
//! - **There is nothing to keep in step.** An entry rewritten by consolidation
//!   is re-read, so a fact that stopped being about a deadline stops carrying
//!   the signal.
//! - **A low-salience fact is still written, by construction rather than by
//!   rule.** Salience has no write path to gate, so no configuration and no
//!   later change can make it one.
//!
//! ## What a reading is worth, and why that is a scale
//!
//! [`SalienceReading::share`] answers a **ratio**: of the salience information
//! this build can detect, how much does this entry carry? A ratio of two sums of
//! the same quantity is dimensionless and lies in `[0, 1]`, so the term cannot
//! grow with how many signals a deployment happens to be able to detect. The
//! signals **divide one fixed lift** rather than each adding one - which is
//! ACT-R's own answer to the same question, and the answer
//! [`crate::domain::situation`] already gives for the situation cue.
//!
//! A detector that never fires on a store therefore changes no ordering: it
//! scales every entry's share by the same factor, and a common factor cannot
//! reorder anything.
//!
//! The lift that ratio is spent against is
//! [`ActivationWeights::reference_use_lift`] - *exactly what one use at the
//! reference age is worth* - computed from the reinforcement term rather than
//! restated, so it introduces no coefficient of its own. What it states is an
//! equivalence between two signals: "this entry carries everything that makes a
//! fact worth keeping" is worth what "you opened this yesterday" is worth. An
//! equivalence transfers to a store nobody measured.
//!
//! **Why not more than that**, when a person's own instruction is the strongest
//! evidence here: a mark in the use log is a record of something that happened,
//! and a salience signal is a reading of what text means. A reading must not
//! outweigh a record. So the whole reading is bounded by one recorded use, and
//! it reorders a bunched block rather than overturning a semantic lead.
//!
//! [`ActivationWeights::reference_use_lift`]:
//!     crate::domain::activation::ActivationWeights::reference_use_lift
//!
//! ## Why the signals are not weighted equally
//!
//! A person asking for something to be kept is stronger evidence than a body of
//! text mentioning money. The two are separated by **who said it**, and priced
//! by the ratio the use log already declares between a person's mark and the
//! model's - [`UseScoreWeights::model_mark`] and
//! [`UseScoreWeights::person_mark`]. No coefficient is introduced here: a
//! deployment that fits its own mark weights from its own use log moves this
//! term with them.
//!
//! ## The term ranks and never admits
//!
//! Admission to a `[Recall]` block is
//! [`RecallRelevance::clears_bar`](crate::ports::recall::RecallRelevance::clears_bar)
//! on a distance, over a list that arrives nearest-first. That is what lets the
//! block say "and N more entries also matched" and mean it. This term is applied
//! after that test, over the set the bar already admitted, so it permutes the
//! block and can never change its membership.
//!
//! ## The two signals this module does not carry
//!
//! #1127 names five kinds of salience. Three of them are the consequence topics
//! below and one is the explicit instruction. The other two are deliberately
//! absent, and neither is an oversight:
//!
//! - **A correction of something the assistant said** is already recorded, as a
//!   negative mark in the use log (#698). The reinforcement term reads it, and
//!   so does [`crate::domain::replay`]'s contradiction term. Detecting it a
//!   third time from text would count one fact three times.
//! - **Repetition across separate conversations** has nothing to read. Nothing
//!   records that a fact recurred, because recurrence today writes a second
//!   entry rather than reinforcing the first. The signal arrives with the
//!   extraction-time matching that #694 owns, not here.

use std::collections::BTreeSet;

use crate::domain::knowledge::KnowledgeEntry;
use crate::domain::knowledge_use::{MarkSource, UseScoreWeights};

/// The provenance value a person's own promotion of an entry carries.
///
/// The string the `source` column holds, repeated here rather than imported:
/// the storage layer owns the column and the domain must not depend on it. A
/// change to either side that is not made on both shows up as
/// [`SalienceSignal::Instructed`] never firing, which
/// `an_entry_a_person_promoted_carries_the_instruction_signal` pins.
pub const SOURCE_EXPLICIT: &str = "explicit";

/// One kind of evidence that a stored fact deserves attention.
///
/// A closed enum rather than free-form keys, for the reason
/// [`SituationField`](crate::domain::situation::SituationField) is one: the
/// weight of a reading is stated over the set of signals, so the two sides have
/// to agree on what a signal *is*. Adding one is a variant here, its cue below,
/// and nothing else - every rule is stated over [`Self::ALL`] rather than over
/// any particular member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SalienceSignal {
    /// A person deliberately promoted this entry, rather than the assistant
    /// extracting it. The strongest signal here, and the only one that is a
    /// recorded act rather than a reading of text.
    Instructed,
    /// The entry names a date something is wanted by.
    Deadline,
    /// The entry is about money.
    Money,
    /// The entry is about health.
    Health,
    /// The entry records a promise made to somebody else.
    Commitment,
}

impl SalienceSignal {
    /// Every signal this build detects, in the order a reading iterates.
    pub const ALL: [SalienceSignal; 5] = [
        Self::Instructed,
        Self::Deadline,
        Self::Money,
        Self::Health,
        Self::Commitment,
    ];

    /// The name this signal is logged and reported under.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Instructed => "instructed",
            Self::Deadline => "deadline",
            Self::Money => "money",
            Self::Health => "health",
            Self::Commitment => "commitment",
        }
    }

    /// Who this signal comes from, which is what prices it.
    ///
    /// [`MarkSource::Person`] where a person stated it and
    /// [`MarkSource::Model`] where the detector read it out of text - the same
    /// distinction the use log makes between a person's mark and the model's,
    /// and priced by the same two weights.
    pub fn source(self) -> MarkSource {
        match self {
            Self::Instructed => MarkSource::Person,
            Self::Deadline | Self::Money | Self::Health | Self::Commitment => MarkSource::Model,
        }
    }

    /// What one occurrence of this signal is worth, before the ratio.
    ///
    /// Never a number of its own: it is the use log's own price for the source
    /// that stated it.
    fn weight(self, weights: &UseScoreWeights) -> f64 {
        weights.mark_weight(self.source()).max(0.0)
    }

    /// The phrases that say this signal is present.
    ///
    /// Phrases rather than single words wherever the single word is ambiguous.
    /// "due" alone fires on "due to", which is one of the commonest phrases in
    /// English prose and says nothing about a deadline; "cost" fires on the cost
    /// of a database query. A detector that fires on everything carries no
    /// information, which the ratio would then spend a share of the lift on.
    ///
    /// Matched as words rather than as substrings - see [`says`]. That is not
    /// tidiness: "rent" sits inside "current", "tax" inside "syntax", "promised"
    /// inside "compromised" and "euros" inside "neuroscience", so a substring
    /// test would read an ordinary engineering note as being about money and a
    /// promise.
    ///
    /// English only, and stated rather than apologised for. A deployment whose
    /// store is in another language sees these signals never fire, which scales
    /// every entry's share by the same factor and therefore reorders nothing -
    /// the store ranks as it did before this term existed.
    fn cues(self) -> &'static [&'static str] {
        match self {
            // Provenance, not text. `Instructed` is decided by the `source`
            // column, so it has no phrase of its own.
            Self::Instructed => &[],
            Self::Deadline => &[
                "deadline",
                "due date",
                "due on",
                "due by",
                "overdue",
                "expires",
                "expiry",
                "expiration",
                "renewal",
                "renews on",
                "cutoff",
                "no later than",
            ],
            Self::Money => &[
                "invoice",
                "payment",
                "salary",
                "rent",
                "mortgage",
                "tax",
                "refund",
                "budget",
                "deposit",
                "dollars",
                "pounds sterling",
                "euros",
                "bank account",
            ],
            Self::Health => &[
                "doctor",
                "dentist",
                "hospital",
                "clinic",
                "prescription",
                "medication",
                "diagnosis",
                "allergy",
                "allergic",
                "surgery",
                "therapy",
                "symptom",
                "vaccine",
                "blood pressure",
            ],
            Self::Commitment => &[
                "promised",
                "agreed to",
                "i owe",
                "owes me",
                "rsvp",
                "said i would",
                "on my behalf",
                "signed up to",
            ],
        }
    }
}

/// How much of a word a cue may fall short of and still be that word.
///
/// A cue is written in one form and a person writes it in several: one invoice
/// and two invoices, a symptom and the symptoms, promised and promising. An
/// inflection is at most three letters in English (`-s`, `-es`, `-ed`, `-ing`),
/// so a cue followed by that much and then a boundary is the same word, and a
/// cue followed by more is a different one. It is what keeps "tax" off
/// "taxonomy" while leaving it on "taxes".
///
/// A bound rather than a list of endings, because the list is the part that
/// would need maintaining and the bound is the part that does the work.
///
/// **An ending, never a stem.** "promising" is not "promised" plus three
/// letters, so a cue written in one of those forms does not fire on the other.
/// This is a bound on over-matching and not a stemmer, and the cue lists are
/// written in the form a person is likeliest to use.
const MAX_CUE_INFLECTION_CHARS: usize = 3;

/// Whether `haystack` says `cue`, as a word rather than as a run of letters.
///
/// The start of a cue must fall on a word boundary, and its end must fall on one
/// too, give or take an inflection of [`MAX_CUE_INFLECTION_CHARS`].
///
/// **The leading boundary is the half that matters.** "rent" sits inside
/// "current", "tax" inside "syntax", "promised" inside "compromised" and "euros"
/// inside "neuroscience" - every one of those is a cue buried in the middle of
/// an unrelated word, and every one is refused by requiring the character before
/// it to be something other than a letter or a digit.
///
/// `haystack` is already lowercased by [`SalienceReading::read`], and both sides
/// are compared as bytes because every cue is ASCII: a byte index into a UTF-8
/// string is a character boundary wherever a match was found, so slicing at one
/// cannot split a multi-byte character.
fn says(haystack: &str, cue: &str) -> bool {
    haystack.match_indices(cue).any(|(at, _)| {
        let starts_a_word = haystack[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let tail = &haystack[at + cue.len()..];
        let inflection = tail
            .chars()
            .take_while(|c| c.is_alphanumeric())
            .take(MAX_CUE_INFLECTION_CHARS + 1)
            .count();
        starts_a_word && inflection <= MAX_CUE_INFLECTION_CHARS
    })
}

/// The stored text and provenance a reading is taken from.
///
/// Borrowed parts rather than a whole entry, so the daily pass - which holds its
/// own row shape and never builds a [`KnowledgeEntry`] - reads the same rule as
/// the recall block. The same split, and the same reason, as
/// [`SituationSources`](crate::domain::situation::SituationSources).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SalienceSource<'a> {
    /// The entry's provenance, as the `source` column holds it.
    /// [`SOURCE_EXPLICIT`] is what a person's own promotion looks like.
    pub provenance: Option<&'a str>,
    /// The entry's body.
    pub content: &'a str,
    /// Its one-line summary, where it has one. Read as well as the body,
    /// because a summary is what a reader of a long entry actually meets.
    pub summary: Option<&'a str>,
    /// Its tags. A store that tags an entry `health` has said what it is about
    /// more plainly than its prose does.
    pub tags: &'a [String],
}

impl<'a> SalienceSource<'a> {
    /// The parts of `entry` a reading is taken from.
    pub fn of(entry: &'a KnowledgeEntry) -> Self {
        Self {
            provenance: entry.source.as_deref(),
            content: &entry.content,
            summary: entry.summary.as_deref(),
            tags: &entry.tags,
        }
    }
}

/// Which salience signals one entry carries.
///
/// An ordered set, so a reading renders and iterates the same way every time.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SalienceReading(BTreeSet<SalienceSignal>);

impl SalienceReading {
    /// Every signal `source` carries, read with no model call.
    ///
    /// One pass builds the haystack - body, summary and tags, lowercased - and
    /// every text signal is a substring test over it. Provenance is read from
    /// the column rather than from the text.
    pub fn read(source: &SalienceSource<'_>) -> Self {
        let mut haystack = source.content.to_lowercase();
        if let Some(summary) = source.summary {
            haystack.push(' ');
            haystack.push_str(&summary.to_lowercase());
        }
        for tag in source.tags {
            haystack.push(' ');
            haystack.push_str(&tag.to_lowercase());
        }

        let mut carried = BTreeSet::new();
        if source.provenance == Some(SOURCE_EXPLICIT) {
            carried.insert(SalienceSignal::Instructed);
        }
        for signal in SalienceSignal::ALL {
            if signal.cues().iter().any(|cue| says(&haystack, cue)) {
                carried.insert(signal);
            }
        }
        Self(carried)
    }

    /// A reading that carries exactly `signals`.
    pub fn of(signals: impl IntoIterator<Item = SalienceSignal>) -> Self {
        Self(signals.into_iter().collect())
    }

    /// Whether this entry carries `signal`.
    pub fn carries(&self, signal: SalienceSignal) -> bool {
        self.0.contains(&signal)
    }

    /// The signals carried, in [`SalienceSignal::ALL`] order.
    pub fn signals(&self) -> impl Iterator<Item = SalienceSignal> + '_ {
        self.0.iter().copied()
    }

    /// Whether nothing at all was detected, which is the ordinary answer.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Of the salience information this build can detect, how much this entry
    /// carries, in `[0, 1]`.
    ///
    /// The ratio of the weight of the signals carried to the weight of every
    /// signal there is. Three properties follow from the shape rather than from
    /// a clamp:
    ///
    /// - **It cannot grow with the number of signals.** Both halves grow
    ///   together, so the signals divide one fixed lift instead of each adding
    ///   one.
    /// - **An entry carrying nothing scores zero**, so a store this detector
    ///   says nothing about ranks exactly as it ranked before the term existed.
    /// - **It is never negative and never over one**, whatever weights it is
    ///   handed, so the bound is a property of this function rather than of its
    ///   caller.
    pub fn share(&self, weights: &UseScoreWeights) -> f64 {
        let detectable: f64 = SalienceSignal::ALL
            .iter()
            .map(|signal| signal.weight(weights))
            .sum();
        if detectable <= 0.0 || !detectable.is_finite() {
            return 0.0;
        }
        let carried: f64 = self.0.iter().map(|signal| signal.weight(weights)).sum();
        (carried / detectable).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An entry whose body is `content`, with nothing else to read.
    fn body(content: &str) -> SalienceReading {
        SalienceReading::read(&SalienceSource {
            content,
            ..SalienceSource::default()
        })
    }

    /// The text that fires each signal, and it is prose a person would write
    /// rather than the cue phrase on its own.
    fn a_fact_carrying(signal: SalienceSignal) -> &'static str {
        match signal {
            SalienceSignal::Instructed => "",
            SalienceSignal::Deadline => "The passport renewal is due by the end of March.",
            SalienceSignal::Money => "The rent went up to twelve hundred a month in April.",
            SalienceSignal::Health => "The doctor moved the follow-up to a Tuesday.",
            SalienceSignal::Commitment => "I promised Sam a draft of the plan before the weekend.",
        }
    }

    /// Acceptance (#1127): every salience signal is detected from what the store
    /// already holds, with no model call.
    ///
    /// Arithmetic and substring tests over the entry's own text and provenance:
    /// no network, no clock, no model. Every signal is covered, so a variant
    /// added without a cue fails here rather than silently never firing.
    #[test]
    fn every_salience_signal_is_detected_from_stored_text_with_no_model_call() {
        for signal in SalienceSignal::ALL {
            let source = SalienceSource {
                provenance: (signal == SalienceSignal::Instructed).then_some(SOURCE_EXPLICIT),
                content: a_fact_carrying(signal),
                ..SalienceSource::default()
            };
            let reading = SalienceReading::read(&source);
            assert!(
                reading.carries(signal),
                "{} was not detected in {:?}",
                signal.as_str(),
                source.content
            );
            assert_eq!(
                reading,
                SalienceReading::read(&source),
                "a reading must be a pure function of what it is given"
            );
        }
    }

    /// A person's own promotion of an entry is the instruction signal, read off
    /// the provenance column rather than guessed from prose.
    #[test]
    fn an_entry_a_person_promoted_carries_the_instruction_signal() {
        let promoted = SalienceReading::read(&SalienceSource {
            provenance: Some(SOURCE_EXPLICIT),
            content: "The spare key is under the third flowerpot.",
            ..SalienceSource::default()
        });
        assert!(promoted.carries(SalienceSignal::Instructed));

        for extracted in ["extraction", "consolidation"] {
            let reading = SalienceReading::read(&SalienceSource {
                provenance: Some(extracted),
                content: "The spare key is under the third flowerpot.",
                ..SalienceSource::default()
            });
            assert!(
                !reading.carries(SalienceSignal::Instructed),
                "{extracted} is the assistant's own doing, not a person's"
            );
        }
    }

    /// A tag says what an entry is about more plainly than its prose does, and a
    /// summary is what a reader of a long entry actually meets. Both are read.
    #[test]
    fn a_signal_in_the_tags_or_the_summary_is_detected_as_well_as_one_in_the_body() {
        let tags = vec!["health".to_string()];
        let tagged = SalienceReading::read(&SalienceSource {
            content: "Tuesdays are no good from now on.",
            tags: &tags,
            ..SalienceSource::default()
        });
        assert!(
            !tagged.carries(SalienceSignal::Health),
            "precondition: the body alone says nothing about health"
        );

        let with_prescription = SalienceReading::read(&SalienceSource {
            content: "Tuesdays are no good from now on.",
            summary: Some("The prescription collection moved."),
            ..SalienceSource::default()
        });
        assert!(with_prescription.carries(SalienceSignal::Health));
    }

    /// A cue is a word, not a run of letters. Every cue this build ships is
    /// checked against a longer word it is buried inside, so nothing here rests
    /// on the handful a reviewer happened to think of.
    #[test]
    fn no_cue_fires_when_it_is_buried_inside_a_longer_word() {
        for signal in SalienceSignal::ALL {
            for cue in signal.cues() {
                let buried = format!("The zz{cue}zzzz of it is not the point.");
                assert!(
                    !body(&buried).carries(signal),
                    "{buried:?} fired {}",
                    signal.as_str()
                );
            }
        }
    }

    /// The collisions a real store produces, named rather than generated, so a
    /// cue added later that reintroduces one fails here with the sentence in the
    /// message.
    ///
    /// Every line is ordinary prose from an engineering note, and every line
    /// contains a shipped cue as a substring: "rent" in "current", "tax" in
    /// "syntax" and "taxonomy", "promised" in "compromised", "euros" in
    /// "neuroscience", and "due" in "due to". "cost" appears too, and is not a
    /// cue for exactly this reason.
    #[test]
    fn ordinary_engineering_prose_carries_no_salience_signal() {
        for innocent in [
            "The build is slow due to the linker step.",
            "The query cost went up after the index was dropped.",
            "The current parser is different from the one the parent crate uses.",
            "The syntax of the taxonomy file changed.",
            "A compromised token is rotated, not repaired.",
            "The neuroscience paper is filed under reading.",
            "The deployment is committed to main once the gate is green.",
        ] {
            let reading = body(innocent);
            assert!(
                reading.is_empty(),
                "{innocent:?} read as carrying {:?}",
                reading.signals().map(|s| s.as_str()).collect::<Vec<_>>()
            );
        }
    }

    /// One invoice and two invoices are the same word, so a cue written in one
    /// form still fires where the text only adds an ending to it.
    ///
    /// Only an ending. A form that changes the stem - "promising" against the
    /// cue "promised" - is a different string and is not detected, which is the
    /// limit [`MAX_CUE_INFLECTION_CHARS`] states and this pins so that nobody
    /// reads the rule as stemming.
    #[test]
    fn a_cue_fires_where_the_text_only_adds_an_ending_to_it() {
        assert!(body("Two invoices arrived.").carries(SalienceSignal::Money));
        assert!(body("The symptoms come and go.").carries(SalienceSignal::Health));
        assert!(body("Both deadlines moved.").carries(SalienceSignal::Deadline));
        assert!(body("She promised a draft.").carries(SalienceSignal::Commitment));

        assert!(
            !body("She is promising a draft.").carries(SalienceSignal::Commitment),
            "the rule adds endings and does not stem, and a test that let this pass would be \
             claiming otherwise"
        );
    }

    /// Case is not a signal. The same fact typed in any case reads the same.
    #[test]
    fn a_reading_is_the_same_reading_however_the_text_is_cased() {
        assert_eq!(
            body("The DENTIST appointment is on a Thursday."),
            body("the dentist appointment is on a thursday.")
        );
    }

    /// Acceptance (#1127): a low-salience fact is written and read like any
    /// other. Nothing gates on this reading, and an empty one contributes
    /// exactly nothing.
    #[test]
    fn an_entry_carrying_no_signal_reads_as_empty_and_shares_nothing() {
        let weights = UseScoreWeights::default();
        let plain = body("The kitchen tap turns the wrong way.");
        assert!(plain.is_empty());
        assert_eq!(plain.signals().count(), 0);
        assert_eq!(plain.share(&weights), 0.0);
    }

    /// Acceptance (#1127): the reading's size cannot grow with how many signals
    /// a build detects, because the signals **divide one fixed share** rather
    /// than each adding one.
    ///
    /// Stated as the arithmetic that makes it true: what every signal is worth
    /// on its own sums to exactly the whole share, so a sixth signal takes from
    /// the five rather than adding a sixth part. An entry carrying everything
    /// reaches exactly one however many signals there are.
    #[test]
    fn the_salience_signals_divide_one_fixed_share_rather_than_each_adding_one() {
        let weights = UseScoreWeights::default();

        let each: f64 = SalienceSignal::ALL
            .iter()
            .map(|signal| SalienceReading::of([*signal]).share(&weights))
            .sum();
        assert!(
            (each - 1.0).abs() < 1e-9,
            "the signals are worth {each} between them, and a share that does not sum to one \
             grows with how many signals there are"
        );

        assert!(
            (SalienceReading::of(SalienceSignal::ALL).share(&weights) - 1.0).abs() < 1e-9,
            "an entry carrying every signal must reach exactly the whole share"
        );
    }

    /// A person saying so outweighs the detector reading it out of prose, and by
    /// the ratio the use log already declares between a person's mark and the
    /// model's rather than by a number of this term's own.
    #[test]
    fn a_person_stated_signal_outweighs_one_read_out_of_text() {
        let weights = UseScoreWeights::default();
        let asked_for = SalienceReading::of([SalienceSignal::Instructed]).share(&weights);
        let about_money = SalienceReading::of([SalienceSignal::Money]).share(&weights);

        assert!(asked_for > about_money);
        let ratio = asked_for / about_money;
        let declared = weights.person_mark / weights.model_mark;
        assert!(
            (ratio - declared).abs() < 1e-9,
            "the two are separated by {ratio}, and the use log's own person-to-model ratio is \
             {declared}"
        );
    }

    /// The share stays in `[0, 1]` over every combination of signals there is -
    /// all thirty-two of them, rather than a favourable few - so the bound is a
    /// property of the function and not of the readings a store happens to
    /// produce.
    #[test]
    fn the_share_stays_in_its_range_over_every_combination_of_signals() {
        let weights = UseScoreWeights::default();
        for mask in 0u32..(1 << SalienceSignal::ALL.len()) {
            let carried: Vec<SalienceSignal> = SalienceSignal::ALL
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, signal)| *signal)
                .collect();
            let share = SalienceReading::of(carried.clone()).share(&weights);
            assert!(
                (0.0..=1.0).contains(&share),
                "{carried:?} shared {share}, outside the range the term is bounded to"
            );
        }
    }

    /// Weights a deployment fitted for itself cannot take the share out of its
    /// range, including the degenerate ones nothing constructs today.
    #[test]
    fn weights_that_price_no_signal_at_all_produce_no_share_rather_than_a_nonsense_one() {
        let nothing_counts = UseScoreWeights {
            model_mark: 0.0,
            person_mark: 0.0,
            ..UseScoreWeights::default()
        };
        assert_eq!(
            SalienceReading::of(SalienceSignal::ALL).share(&nothing_counts),
            0.0
        );

        let negative = UseScoreWeights {
            model_mark: -3.0,
            person_mark: -3.0,
            ..UseScoreWeights::default()
        };
        let share = SalienceReading::of([SalienceSignal::Money]).share(&negative);
        assert!(
            (0.0..=1.0).contains(&share),
            "a negative mark weight shared {share}"
        );
    }

    /// Stored names are stable, because they are what a reading is logged and
    /// reported under.
    #[test]
    fn every_signal_has_its_own_stable_name() {
        let mut names: Vec<&str> = SalienceSignal::ALL
            .iter()
            .map(|signal| signal.as_str())
            .collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two signals share a name");
    }
}
