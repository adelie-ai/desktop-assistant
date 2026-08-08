//! What makes one stored fact worth more attention than another (#1127).
//!
//! Emotionally significant events consolidate preferentially in people. The
//! software analogues are cheap: an entry with a date on it, an entry about
//! money, health, or a promise made to somebody else, and an entry somebody
//! wrote in a live turn rather than one the dream cycle distilled overnight.
//! None of them needs a model call, because each is a property of text and
//! provenance the store already holds.
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
//! [`SalienceReading::share`] answers a **ratio**: of the signals this build can
//! detect, how many does this entry carry? A ratio of two counts of the same
//! thing is dimensionless and lies in `[0, 1]`, so the term cannot grow with how
//! many signals a deployment happens to be able to detect. The signals **divide
//! one fixed lift** rather than each adding one - which is ACT-R's own answer to
//! the same question, and the answer [`crate::domain::situation`] already gives
//! for the situation cue.
//!
//! The lift that ratio is spent against is
//! [`ActivationWeights::reference_use_lift`] - *exactly what one use at the
//! reference age is worth* - computed from the reinforcement term rather than
//! restated, so it introduces no coefficient of its own. What it states is an
//! equivalence between two signals: "this entry carries everything that makes a
//! fact worth keeping" is worth what "you opened this yesterday" is worth. An
//! equivalence transfers to a store nobody measured.
//!
//! **Why not more than that**, when one of the signals may be a person's own
//! doing: a mark in the use log records something that happened, and a salience
//! signal reads what text means. A reading must not outweigh a record.
//!
//! **Every signal is worth the same.** That is not a claim that a deadline and a
//! live-turn write are equally strong evidence - they are not. It is a refusal
//! to invent a number that says how much stronger, because nothing in this store
//! measures it. An equal split is the honest reading of five signals nobody has
//! weighed, and a deployment that later measures its own can weigh them then.
//!
//! **Adding a detector is a change to the ranking, not a free extension.** A
//! signal that never fires on a store cannot reorder two entries *against each
//! other on salience*, because it scales both shares by the same factor. It does
//! shrink the whole term against the semantic and use-log terms beside it, so a
//! pair that salience was separating by a hair can change places. The bound
//! holds; the order is not promised.
//!
//! [`ActivationWeights::reference_use_lift`]:
//!     crate::domain::activation::ActivationWeights::reference_use_lift
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
//! ## What this module does not detect, and says so
//!
//! - **A correction of something the assistant said** is already recorded, as a
//!   negative mark in the use log (#698). The reinforcement term reads it, and
//!   so does [`crate::domain::replay`]'s contradiction term. Detecting it a
//!   third time from text would count one fact three times.
//! - **Repetition across separate conversations** has nothing to read. Nothing
//!   records that a fact recurred, because recurrence today writes a second
//!   entry rather than reinforcing the first. The signal arrives with the
//!   extraction-time matching that #694 owns, not here.
//! - **How near a deadline is.** [`SalienceSignal::Deadline`] fires on an entry
//!   that names a date something is wanted by, and no date is parsed, so a
//!   deadline three years past reads exactly like one due tomorrow. Proximity
//!   needs a parsed date and a decay, which is a larger thing than a phrase
//!   list; presence is what this build measures and presence is what the term
//!   is worth.
//! - **Anything not written in English.** Every cue below is an English phrase.
//!   A store in another language carries the text signals on no entry at all -
//!   though [`SalienceSignal::Deliberate`] still fires, because it is read from
//!   a column rather than from prose.

use std::collections::BTreeSet;

use crate::domain::knowledge::KnowledgeEntry;

/// The provenance value an entry written during a live turn carries.
///
/// The string the `source` column holds, defined here rather than in storage
/// because the domain must not depend on the storage layer. Storage names the
/// same constant rather than declaring a second one, so the two cannot drift:
/// a drift would show up only as a signal that never fires, which is the
/// quietest failure a ranking term has.
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
    /// The entry was written during a live turn rather than distilled by the
    /// dream cycle: [`SOURCE_EXPLICIT`] provenance.
    ///
    /// **It does not say a person asked for it**, and the name is `Deliberate`
    /// rather than anything stronger for that reason. The column records that
    /// somebody was there and the write was an act rather than a batch
    /// inference - the person asked, or the assistant decided in the moment -
    /// and it cannot separate those two. A signal priced on the stronger reading
    /// would put half the term on every tool-written row in the store.
    Deliberate,
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
        Self::Deliberate,
        Self::Deadline,
        Self::Money,
        Self::Health,
        Self::Commitment,
    ];

    /// A short distinct name for this signal, for a message that has to say
    /// which one. Nothing stores it - see "Read, never written" above.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deliberate => "deliberate",
            Self::Deadline => "deadline",
            Self::Money => "money",
            Self::Health => "health",
            Self::Commitment => "commitment",
        }
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
    /// **No bare topic word is a cue.** "health" would be one on the face of
    /// it - a tag is part of the haystack, and a store that tags an entry
    /// `health` has said what it is about. But this design prices every cue the
    /// same, and a topic word is weak evidence where a specific one is strong,
    /// so while they are worth alike a topic word can only add noise. "health
    /// check", "service health" and "healthy" are ordinary in the engineering
    /// notes this store holds, and none of them is about anybody's health. A
    /// bare topic word wants a lower weight than a specific cue, and that is a
    /// design change rather than a longer list.
    fn cues(self) -> &'static [&'static str] {
        match self {
            // Provenance, not text. `Deliberate` is decided by the `source`
            // column, so it has no phrase of its own.
            Self::Deliberate => &[],
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
/// and two invoices, a symptom and the symptoms. An inflection is at most three
/// letters in English (`-s`, `-es`, `-ed`, `-ing`), so a cue followed by that
/// much and then a boundary is the same word, and a cue followed by more is a
/// different one. It is what keeps "tax" off "taxonomy" while leaving it on
/// "taxes".
///
/// A bound rather than a list of endings, because the list is the part that
/// would need maintaining and the bound is the part that does the work.
///
/// **An ending, never a stem.** "promising" is not "promised" plus three
/// letters, so a cue written in one of those forms does not fire on the other.
/// This is a bound on over-matching and not a stemmer, and the cue lists are
/// written in the form a person is likeliest to use.
const MAX_CUE_INFLECTION_CHARS: usize = 3;

/// What separates the entry's body, its summary and each of its tags in the
/// haystack a reading is read from.
///
/// A newline rather than a space, and it is load-bearing rather than cosmetic.
/// Cues are phrases whose words are separated by a space, and a space here would
/// let one straddle two fields that each say nothing: an entry tagged `bank` and
/// `account` would carry the phrase "bank account" that neither tag states, and
/// a body ending "...is due" beside a summary opening "by Friday" would carry
/// "due by". With a newline the phrase is simply not in the haystack, so the
/// match never happens and no boundary test is involved.
///
/// It has to be non-alphanumeric for the other half of the job: a single-word
/// cue at the start of a tag is preceded by this character, and [`says`] admits
/// a match only where what precedes it is not a letter or a digit.
const FIELD_SEPARATOR: char = '\n';

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
        let inflection = haystack[at + cue.len()..]
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
    /// [`SOURCE_EXPLICIT`] is what a live-turn write looks like.
    pub provenance: Option<&'a str>,
    /// The entry's body.
    pub content: &'a str,
    /// Its one-line summary, where it has one. Read as well as the body,
    /// because a summary is what a reader of a long entry actually meets.
    pub summary: Option<&'a str>,
    /// Its tags.
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
    /// One pass builds the haystack - body, summary and tags, lowercased and
    /// separated by a newline - and every text signal is a word test over it.
    /// Provenance is read from the column rather than from the text.
    ///
    /// The separator is load-bearing rather than cosmetic: cues are phrases, and
    /// a space would let one straddle two fields that each say nothing. An entry
    /// tagged `bank` and `account` would carry a phrase neither tag states.
    pub fn read(source: &SalienceSource<'_>) -> Self {
        let mut haystack = source.content.to_lowercase();
        for field in source
            .summary
            .into_iter()
            .chain(source.tags.iter().map(String::as_str))
        {
            haystack.push(FIELD_SEPARATOR);
            haystack.push_str(&field.to_lowercase());
        }

        let mut carried = BTreeSet::new();
        if source.provenance == Some(SOURCE_EXPLICIT) {
            carried.insert(SalienceSignal::Deliberate);
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

    /// Of the signals this build can detect, how many this entry carries, in
    /// `[0, 1]`.
    ///
    /// Three properties follow from the shape rather than from a clamp:
    ///
    /// - **It cannot grow with the number of signals.** Both halves grow
    ///   together, so the signals divide one fixed lift instead of each adding
    ///   one.
    /// - **An entry carrying nothing scores zero**, so a store this detector
    ///   says nothing about ranks exactly as it ranked before the term existed.
    /// - **It is never negative and never over one**, so the bound is a property
    ///   of this function rather than of its caller.
    pub fn share(&self) -> f64 {
        // The set can only hold signals the enum declares, so the ratio cannot
        // pass one; the clamp says so rather than leaving a reader to work it
        // out from the type.
        (self.0.len() as f64 / SalienceSignal::ALL.len() as f64).clamp(0.0, 1.0)
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
            SalienceSignal::Deliberate => "",
            SalienceSignal::Deadline => "The passport renewal is due by the end of March.",
            SalienceSignal::Money => "The rent went up to twelve hundred a month in April.",
            SalienceSignal::Health => "The doctor moved the follow-up to a Tuesday.",
            SalienceSignal::Commitment => "I promised Sam a draft of the plan before the weekend.",
        }
    }

    /// Acceptance (#1127): every salience signal is detected from what the store
    /// already holds, with no model call.
    ///
    /// Arithmetic and word tests over the entry's own text and provenance: no
    /// network, no clock, no model. Every signal is covered, so a variant added
    /// without a cue fails here rather than silently never firing.
    #[test]
    fn every_salience_signal_is_detected_from_stored_text_with_no_model_call() {
        for signal in SalienceSignal::ALL {
            let source = SalienceSource {
                provenance: (signal == SalienceSignal::Deliberate).then_some(SOURCE_EXPLICIT),
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

    /// An entry written during a live turn carries the deliberate signal, read
    /// off the provenance column rather than guessed from prose.
    ///
    /// And what the column cannot say: the same signal fires whoever did the
    /// writing, because `explicit` covers both a person asking and the assistant
    /// deciding in the moment. Pinned so nobody prices it as a person's word.
    #[test]
    fn an_entry_written_in_a_live_turn_carries_the_deliberate_signal() {
        let live = SalienceReading::read(&SalienceSource {
            provenance: Some(SOURCE_EXPLICIT),
            content: "The spare key is under the third flowerpot.",
            ..SalienceSource::default()
        });
        assert!(live.carries(SalienceSignal::Deliberate));

        for distilled in ["extraction", "consolidation"] {
            let reading = SalienceReading::read(&SalienceSource {
                provenance: Some(distilled),
                content: "The spare key is under the third flowerpot.",
                ..SalienceSource::default()
            });
            assert!(
                !reading.carries(SalienceSignal::Deliberate),
                "{distilled} is the dream cycle's doing, so no live turn wrote it"
            );
        }

        assert!(
            !SalienceReading::read(&SalienceSource {
                content: "The spare key is under the third flowerpot.",
                ..SalienceSource::default()
            })
            .carries(SalienceSignal::Deliberate),
            "a row written before the column existed says nothing either way"
        );
    }

    /// The tags and the summary are read as well as the body, and either can
    /// fire a signal the body does not.
    ///
    /// The tag here is a cue rather than a bare topic word, because a bare topic
    /// word is deliberately not one - `cues` says why. An entry tagged only
    /// `health` carries no signal, and that is the cost of pricing every cue
    /// alike, stated here so it is met as a decision rather than as a surprise.
    #[test]
    fn a_signal_in_the_tags_or_the_summary_fires_where_the_body_alone_would_not() {
        let plain = "Tuesdays are no good from now on.";
        assert!(
            body(plain).is_empty(),
            "precondition: the body alone carries nothing"
        );

        let tags = vec!["invoice".to_string()];
        assert!(
            SalienceReading::read(&SalienceSource {
                content: plain,
                tags: &tags,
                ..SalienceSource::default()
            })
            .carries(SalienceSignal::Money),
            "a tag that is a cue must reach the reading"
        );

        assert!(
            SalienceReading::read(&SalienceSource {
                content: plain,
                summary: Some("The prescription collection moved."),
                ..SalienceSource::default()
            })
            .carries(SalienceSignal::Health),
            "a summary must reach the reading"
        );

        let topic_only = vec!["health".to_string()];
        assert!(
            SalienceReading::read(&SalienceSource {
                content: plain,
                tags: &topic_only,
                ..SalienceSource::default()
            })
            .is_empty(),
            "a bare topic word is not a cue, deliberately"
        );
    }

    /// A phrase cue may not be assembled out of two fields that each say
    /// nothing. The fields are joined by a separator no cue can cross.
    #[test]
    fn a_phrase_cue_does_not_fire_across_two_fields_that_each_say_nothing() {
        let tags = vec!["bank".to_string(), "account".to_string()];
        assert!(
            !SalienceReading::read(&SalienceSource {
                content: "Two words that only mean something together.",
                tags: &tags,
                ..SalienceSource::default()
            })
            .carries(SalienceSignal::Money),
            "\"bank\" and \"account\" are two tags, not the phrase \"bank account\""
        );

        assert!(
            !SalienceReading::read(&SalienceSource {
                content: "The report is due",
                summary: Some("by Friday the team reconvenes"),
                ..SalienceSource::default()
            })
            .carries(SalienceSignal::Deadline),
            "\"due\" ending a body and \"by\" opening a summary are not \"due by\""
        );
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
    /// contains a shipped cue as a substring: "rent" in "current", "different"
    /// and "parent", "tax" in "syntax" and "taxonomy", "promised" in
    /// "compromised", "euros" in "neuroscience", "i owe" in "Naomi owed", and
    /// "due" in "due to". "cost" appears too, and is not a cue for exactly this
    /// reason.
    #[test]
    fn ordinary_engineering_prose_carries_no_salience_signal() {
        for innocent in [
            "The build is slow due to the linker step.",
            "The query cost went up after the index was dropped.",
            "The current parser is different from the one the parent crate uses.",
            "The syntax of the taxonomy file changed.",
            "A compromised token is rotated, not repaired.",
            "The neuroscience paper is filed under reading.",
            "Naomi owed the team a review of the retry loop.",
            "The deployment is committed to main once the gate is green.",
            "The health check endpoint returns 200.",
            "A medical-imaging customer asked about the money-transfer flow.",
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

    /// Acceptance (#1127): a low-salience fact is read like any other. Nothing
    /// gates on this reading, and an empty one contributes exactly nothing.
    #[test]
    fn an_entry_carrying_no_signal_reads_as_empty_and_shares_nothing() {
        let plain = body("The kitchen tap turns the wrong way.");
        assert!(plain.is_empty());
        assert_eq!(plain.signals().count(), 0);
        assert_eq!(plain.share(), 0.0);
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
        let each: f64 = SalienceSignal::ALL
            .iter()
            .map(|signal| SalienceReading::of([*signal]).share())
            .sum();
        assert!(
            (each - 1.0).abs() < 1e-9,
            "the signals are worth {each} between them, and a share that does not sum to one \
             grows with how many signals there are"
        );

        assert!(
            (SalienceReading::of(SalienceSignal::ALL).share() - 1.0).abs() < 1e-9,
            "an entry carrying every signal must reach exactly the whole share"
        );
    }

    /// Every signal is worth the same, deliberately: nothing in this store
    /// measures how much stronger one is than another, so no number here says.
    #[test]
    fn no_signal_is_worth_more_than_another() {
        let first = SalienceReading::of([SalienceSignal::ALL[0]]).share();
        for signal in SalienceSignal::ALL {
            assert_eq!(
                SalienceReading::of([signal]).share(),
                first,
                "{} is priced differently, and nothing measures that difference",
                signal.as_str()
            );
        }
    }

    /// The share stays in `[0, 1]` over every combination of signals there is -
    /// all thirty-two of them, rather than a favourable few - so the bound is a
    /// property of the function and not of the readings a store happens to
    /// produce.
    #[test]
    fn the_share_stays_in_its_range_over_every_combination_of_signals() {
        for mask in 0u32..(1 << SalienceSignal::ALL.len()) {
            let carried: Vec<SalienceSignal> = SalienceSignal::ALL
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, signal)| *signal)
                .collect();
            let share = SalienceReading::of(carried.clone()).share();
            assert!(
                (0.0..=1.0).contains(&share),
                "{carried:?} shared {share}, outside the range the term is bounded to"
            );
        }
    }

    /// Every signal is named, and no two share a name, so a message that says
    /// which signal fired says one thing.
    #[test]
    fn every_signal_has_a_name_of_its_own() {
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
