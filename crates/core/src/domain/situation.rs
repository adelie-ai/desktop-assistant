//! The situation as a retrieval cue (#1125, absorbing #238).
//!
//! Retrieval is keyed on prompt text. For a life assistant that is the weakest
//! cue available: people describe life events vaguely, and almost never in the
//! words an entry was written with. Encoding specificity says recall depends on
//! the overlap between the cue present when a memory was written and the cue
//! present when it is sought - which is why walking into a room brings back what
//! was thought there. The situation is the strongest cue a desktop assistant
//! holds, and it is free, because the system already knows it without asking.
//!
//! This module is the whole rule. It says what a situation is, what the present
//! one is worth as a cue, and how much of that an entry's own record accounts
//! for. [`crate::domain::activation`] turns the answer into the third term of
//! `A_i`.
//!
//! ## Three quantities, and they are not the same thing
//!
//! - A [`Situation`] is **the present**: at most one value per field, derived
//!   from whatever sources are connected. Every field is optional, and a
//!   deployment with nothing connected produces an empty one.
//! - A [`SituationRecord`] is **an entry's own history**: the values it has been
//!   seen in, possibly several per field. An entry acquires one when it is
//!   written and adds to it every time it proves useful somewhere new, which is
//!   #238's accumulation rule.
//! - A [`SituationCue`] is **the present read against the store**: the same
//!   [`Situation`], plus how much each of its values separates one entry of this
//!   store from another. Only the store can say that, so the adapter measures it
//!   and the core reads it - the same split [`RecallDispersion`] already makes
//!   for distance.
//!
//! [`RecallDispersion`]: crate::ports::recall::RecallDispersion
//!
//! ## Why the cue is weighted by what it separates
//!
//! A plain overlap fraction has a defect that only shows on a real store: most
//! deployments have exactly one host. Every entry then carries `host=<the only
//! host>`, every prompt matches it, and the term becomes a constant added to
//! every entry that happens to have a record - which reorders nothing among
//! them, and sinks every entry written before this feature shipped. The cue
//! would be measuring when the code landed rather than where the person is.
//!
//! So each cue value is weighted by its own self-information in this store,
//! `ln(population / fan)`: how much knowing the value narrows the field.
//!
//! - A value every entry carries has `fan = population`, so it is worth **zero
//!   nats**. Your only host tells nobody anything.
//! - A value one entry carries is worth `ln(population)` nats, the most this
//!   store can offer.
//! - A value **no** entry carries is worth zero as well, and for the same
//!   reason rather than a different one: a field on which no candidate can match
//!   separates nobody. It is the one point the formula does not reach - `fan`
//!   of zero is not in its domain - and it is stated rather than clamped,
//!   because the reasoning is the reasoning and not an edge case.
//!
//! **Both counts are per field**, and that is load-bearing rather than tidy -
//! see [`FieldFan`]. Which fields an observation can read depends on the client
//! that made it, so a store's coverage is uneven: a host may sit on a third of
//! the entries while the weekday sits on all of them. Divided by one store-wide
//! count, the only host in a store would come out informative merely because
//! two thirds of the entries record no host at all, which is the very error the
//! weight exists to prevent.
//!
//! This is Anderson's fan effect, and it arrives here as the definition of the
//! weight rather than as a correction bolted onto one.
//!
//! ## What the term is worth, and why that is a scale
//!
//! [`SituationCue::coverage`] answers a **ratio**: of the information the
//! present cue carries about the entries this one could have matched, how much
//! did it account for? A ratio of two sums of the same quantity is
//! dimensionless and lies in `[0, 1]`, so the term cannot grow with how many
//! situation fields a deployment has connected. That is ACT-R's own answer to
//! the same question: the source activation available to spread is fixed and
//! divided among the cues present, rather than accumulated over them.
//!
//! The lift that ratio is spent against is
//! [`ActivationWeights::situation_lift`], which is defined as *exactly what one
//! use at the reference age is worth* and computed from the reinforcement term
//! rather than restated. It introduces no coefficient of its own:
//!
//! - It states an **equivalence between two signals** - "this entry recurs where
//!   you are now" is worth what "you opened this yesterday" is worth - rather
//!   than a value read off one store's observed spread, which is the compromise
//!   [`DEFAULT_USE_LIFT`] had to make.
//! - It carries **no unit**. Count the use log's ages in hours instead of
//!   seconds and the lift is unchanged, because the reinforcement term is
//!   already a ratio against its own reference. It follows
//!   [`USE_REFERENCE_AGE_SECONDS`], which is a genuine unit normalization, and
//!   not [`DEFAULT_USE_LIFT`], which is not.
//! - A deployment that fits its own `use_lift` from its own use log moves both
//!   terms together and keeps the stated relation, so there is never a second
//!   number to fit.
//!
//! At the default weights that bound is about a third of a deviation: enough to
//! settle a near-tie, and a ninth of [`MAX_REINFORCEMENT_DEVIATIONS`], so the
//! situation can reorder a bunched block and can never overturn a semantic
//! lead. Its influence is largest exactly where the whole admitted band is
//! narrow, which is the weakly cued prompt - the case #1123 settled should let
//! the other signals lead, because a best match sitting just above the bar means
//! the prompt named nothing the store really holds.
//!
//! [`DEFAULT_USE_LIFT`]: crate::domain::activation::DEFAULT_USE_LIFT
//! [`USE_REFERENCE_AGE_SECONDS`]: crate::domain::activation::USE_REFERENCE_AGE_SECONDS
//! [`MAX_REINFORCEMENT_DEVIATIONS`]: crate::domain::activation::MAX_REINFORCEMENT_DEVIATIONS
//! [`ActivationWeights::situation_lift`]:
//!     crate::domain::activation::ActivationWeights::situation_lift
//!
//! ## The term ranks and never admits
//!
//! Admission is [`RecallRelevance::clears_bar`] on a distance, over a list that
//! arrives nearest-first. That is what lets the block say "and N more entries
//! also matched" and mean it. This term is applied after that test, over the set
//! the bar already admitted, so it permutes the block and can never change its
//! membership. Nothing here is reachable from the admission path, which is the
//! guarantee rather than a convention.
//!
//! [`RecallRelevance::clears_bar`]: crate::ports::recall::RecallRelevance::clears_bar
//!
//! ## Presence is the match, not how often
//!
//! A record holds how many times a value has been seen and when it was last
//! seen, and the match reads neither. Two reasons, and the second is the one
//! that matters:
//!
//! - The use log already measures how much an entry has been used. Weighting a
//!   situation match by the same count would put that signal into `A_i` twice.
//! - It closes the feedback loop. An entry that ranks up in a situation gets
//!   opened there, which records the situation, which would raise it further. A
//!   binary match ends that after one step: the second recording of a value the
//!   record already holds changes nothing at all.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Datelike, Timelike, Utc};

/// How many entries a store must hold records for **on one field** before that
/// field's fan is read as information.
///
/// A fan measured over a handful of entries is noise, and noise in the weight
/// makes the ratio meaningless. The same floor, and the same reason, as
/// [`RECALL_DISPERSION_MIN_ROWS`]: it is a sample-size floor rather than a
/// coefficient, so it says when a measurement may be believed and never how much
/// the measurement is worth. A field below it is weighted at zero, and a cue
/// whose every field is below it is no cue at all - every entry then ranks the
/// way it ranked before this term existed.
///
/// **Per field, not per store**, because the fan is per field. Which fields an
/// observation could read depends on what the client that made it reported, so
/// two clients give a store uneven coverage: a host may be recorded on a third
/// of the entries while the weekday is on all of them. Measuring both against
/// one store-wide count would read the host's absence from two thirds of the
/// store as evidence that the host is informative, when among the entries that
/// have one it may separate nobody.
///
/// [`RECALL_DISPERSION_MIN_ROWS`]: crate::ports::recall::RECALL_DISPERSION_MIN_ROWS
pub const SITUATION_MIN_POPULATION: u64 = 20;

/// How long one situation value may be.
///
/// A host arrives from a client's self-reported context and nothing before this
/// point bounds it. That matters more here than it would in a text column,
/// because the value is part of a primary key and part of the fan index: a
/// btree tuple has a hard size limit, so an unbounded value would let a client
/// choose a value the database refuses to index, and the write carrying it would
/// fail rather than the value being odd. Cutting rather than refusing is the
/// trade a mark's reason already makes - an over-long value costs its tail,
/// never the record.
///
/// The figure comes from the longest value a real source produces: a fully
/// qualified domain name is at most 253 characters and a single label at most
/// 63, so this holds any host a person would recognise. Counted in characters
/// and read as bytes it is at most four times this, which leaves the index
/// tuple an order of magnitude inside what a btree accepts.
pub const MAX_SITUATION_VALUE_CHARS: usize = 128;

/// How many values one entry may hold for one field.
///
/// A storage bound, not a ranking coefficient - the same kind of figure as
/// [`MAX_STANDING_OFFERS`], and chosen the same way. Two of the fields below are
/// closed sets already (four parts of a day, seven days of a week), so this
/// binds only an open one such as a host, where an entry that has been useful
/// from eight different machines has stopped saying anything about where it is
/// useful. The writer evicts the least recently seen.
///
/// [`MAX_STANDING_OFFERS`]: crate::ports::knowledge_use::MAX_STANDING_OFFERS
pub const MAX_SITUATION_VALUES_PER_FIELD: usize = 8;

/// One situation value as it is stored: trimmed, cut to
/// [`MAX_SITUATION_VALUE_CHARS`], and with anything a text column cannot hold
/// removed.
///
/// `None` where nothing usable is left, which is the same answer as a field with
/// no source. Two rules, and both exist because the value is a database key
/// rather than a display string:
///
/// - **No NUL byte.** Postgres `text` cannot hold one, so a value carrying one
///   raises on the wire and takes the whole write with it - the trap
///   `storable` already guards for an entry id.
/// - **No newline or control character.** The value is a key a fan is counted
///   over and a name an operator reads back; a control character makes two
///   values that look identical count separately.
fn storable_value(value: &str) -> Option<String> {
    let cleaned: String = value
        .trim()
        .chars()
        .filter(|c| !c.is_control())
        .take(MAX_SITUATION_VALUE_CHARS)
        .collect();
    let cleaned = cleaned.trim().to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// One dimension of a situation.
///
/// Deliberately a closed enum rather than free-form keys. A cue value is
/// compared for equality and counted across a store, so the two sides have to
/// agree on what a field *is*; a string key lets a writer and a reader disagree
/// silently and produce a term that never fires. Adding a dimension is adding a
/// variant here, one arm in [`Situation::observe`], and nothing else - every
/// scoring rule is stated over the set of fields present rather than over any
/// particular one.
///
/// The set is what a desktop assistant can already answer without asking. The
/// dimensions #1125 names that are not here - people present, the current
/// calendar event, the active project, recent topics - have no connected source
/// in this daemon yet, and a field with nothing to read is a column nobody
/// fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SituationField {
    /// The device the person is at, from the client's self-reported hostname
    /// (#549). Free-form: it is whatever the client calls itself.
    Host,
    /// Which part of the local day it is - see [`TimeOfDay`].
    TimeOfDay,
    /// Which day of the local week it is, in English, lowercase (`"monday"`).
    Weekday,
}

impl SituationField {
    /// Every field, in the order a record and a cue iterate.
    pub const ALL: [SituationField; 3] = [Self::Host, Self::TimeOfDay, Self::Weekday];

    /// The name this field is stored and logged under.
    ///
    /// Stable: it is a value in a database column, so renaming one orphans every
    /// row already written under the old name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::TimeOfDay => "time_of_day",
            Self::Weekday => "weekday",
        }
    }

    /// The field a stored name refers to, or `None` for a name no variant
    /// claims.
    ///
    /// `None` is the ordinary answer for a row written by a later version that
    /// knows a field this one does not. Such a row is skipped rather than
    /// refused: an unknown dimension is one this reader cannot score, not a
    /// corrupt record.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|field| field.as_str() == name)
    }
}

/// Which part of the local day it is.
///
/// Four named parts, because they are the parts the language already has and
/// the assistant already speaks. The boundaries are ordinary rather than
/// fitted, and nothing rests on them being the right ones: a deployment whose
/// activity all falls in one part gives that part a fan equal to its whole
/// population, and [`SituationCue`] then weights the field at zero of its own
/// accord.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeOfDay {
    /// 22:00 to 05:00.
    Night,
    /// 05:00 to 12:00.
    Morning,
    /// 12:00 to 17:00.
    Afternoon,
    /// 17:00 to 22:00.
    Evening,
}

impl TimeOfDay {
    /// The part of the day a local hour falls in.
    pub fn at_hour(hour: u32) -> Self {
        match hour {
            5..=11 => Self::Morning,
            12..=16 => Self::Afternoon,
            17..=21 => Self::Evening,
            _ => Self::Night,
        }
    }

    /// The name this part of the day is stored under. Stable, for the reason
    /// [`SituationField::as_str`] gives.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Night => "night",
            Self::Morning => "morning",
            Self::Afternoon => "afternoon",
            Self::Evening => "evening",
        }
    }
}

/// What a deployment has connected, as of one observation.
///
/// Every field is optional and absent is the ordinary answer: which sources
/// exist depends on what the client reported and what this daemon is wired to.
/// A new dimension arrives as a new field here, and every caller keeps compiling
/// through `..Default::default()`.
///
/// Primitives rather than the wire type the values arrive on. What a situation
/// *is* belongs to the domain; which field of which client message feeds it is
/// the adapter's to know.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SituationSources<'a> {
    /// The device the person is at.
    pub host: Option<&'a str>,
    /// The person's IANA timezone (`"Europe/London"`). It gates both clock
    /// fields - see [`Situation::observe`].
    pub timezone: Option<&'a str>,
}

/// The present situation: at most one value per field.
///
/// An empty one is an ordinary answer, and it is the answer on a deployment with
/// nothing connected.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Situation(BTreeMap<SituationField, String>);

impl Situation {
    /// A situation with nothing in it.
    pub fn new() -> Self {
        Self::default()
    }

    /// The same situation with one field stated. Builder-style, for callers
    /// that know a value directly - chiefly tests and later sources.
    #[must_use]
    pub fn with(mut self, field: SituationField, value: impl Into<String>) -> Self {
        let Some(value) = storable_value(&value.into()) else {
            return self;
        };
        self.0.insert(field, value);
        self
    }

    /// Read the situation off the clock and whatever sources are connected.
    ///
    /// Every field is optional and a missing source contributes nothing rather
    /// than a placeholder. Two rules the shape rests on:
    ///
    /// - **Arithmetic only.** A stored timestamp, a timezone name and a
    ///   hostname, and no model, no network, and no clock but the one handed in.
    ///   Capture sits on the write path, and a write path that waits on a model
    ///   is a write path that fails when the model does.
    /// - **The clock fields are gated on a timezone.** A time of day computed in
    ///   the wrong zone is not a missing field, it is a wrong one, and a wrong
    ///   value is worse than an absent value here: absence costs the field, a
    ///   wrong value costs every entry that recorded the same instant honestly.
    ///   An unparseable zone name is treated as an absent one.
    pub fn observe(now: DateTime<Utc>, sources: &SituationSources<'_>) -> Self {
        let mut situation = Self::new();
        if let Some(host) = sources.host {
            // Trimmed and lowercased, because the match is string equality and
            // the value is self-reported. One machine that answers `Workshop`
            // to one client and `workshop` to another would otherwise hold two
            // values, halve its own fan, and match neither prompt fully.
            situation = situation.with(SituationField::Host, host.to_lowercase());
        }
        let Some(local) = sources.timezone.and_then(|zone| local_time(now, zone)) else {
            return situation;
        };
        situation
            .with(
                SituationField::TimeOfDay,
                TimeOfDay::at_hour(local.hour()).as_str(),
            )
            .with(SituationField::Weekday, weekday_name(local))
    }

    /// Whether no field is stated.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The value this situation holds for `field`, if any.
    pub fn get(&self, field: SituationField) -> Option<&str> {
        self.0.get(&field).map(String::as_str)
    }

    /// Every stated field and its value, in [`SituationField`] order.
    pub fn iter(&self) -> impl Iterator<Item = (SituationField, &str)> {
        self.0.iter().map(|(field, value)| (*field, value.as_str()))
    }
}

/// The situations one entry has been seen in.
///
/// Several values per field, because an entry is written once and proves useful
/// many times - #238's accumulation rule. An empty record is the ordinary answer
/// for an entry written before this existed, or written by a path with nothing
/// connected, and it scores exactly zero.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SituationRecord(BTreeMap<SituationField, BTreeSet<String>>);

impl SituationRecord {
    /// A record of nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// The same record with one more value seen for one field.
    #[must_use]
    pub fn with(mut self, field: SituationField, value: impl Into<String>) -> Self {
        let Some(value) = storable_value(&value.into()) else {
            return self;
        };
        self.0.entry(field).or_default().insert(value);
        self
    }

    /// Everything the entry has been seen in, in [`SituationField`] order.
    pub fn from_pairs<I, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (SituationField, V)>,
        V: Into<String>,
    {
        pairs
            .into_iter()
            .fold(Self::new(), |record, (field, value)| {
                record.with(field, value)
            })
    }

    /// Whether this entry has ever been seen with a value for `field`.
    ///
    /// The question that decides whether a field is comparable at all. An entry
    /// that has never been seen with one is not evidence against itself - see
    /// [`SituationCue::coverage`].
    pub fn knows(&self, field: SituationField) -> bool {
        self.0.get(&field).is_some_and(|values| !values.is_empty())
    }

    /// Whether this entry has been seen with exactly this value.
    pub fn holds(&self, field: SituationField, value: &str) -> bool {
        self.0
            .get(&field)
            .is_some_and(|values| values.contains(value))
    }

    /// Whether the entry has been seen in no situation at all.
    pub fn is_empty(&self) -> bool {
        self.0.values().all(BTreeSet::is_empty)
    }

    /// Every field and value pair, in [`SituationField`] order.
    pub fn iter(&self) -> impl Iterator<Item = (SituationField, &str)> {
        self.0
            .iter()
            .flat_map(|(field, values)| values.iter().map(|value| (*field, value.as_str())))
    }
}

/// What one store says about one cue value: how far it narrows the field.
///
/// Both counts are taken over the entries that record **this field**, never over
/// the store as a whole, because that is the population the value is drawn
/// from. `holding / population` is then the probability an entry carries this
/// value given that it says anything about the field at all, and its logarithm
/// is the information the value carries. A store-wide denominator would answer a
/// different question - how many entries record this field - and read a gap in
/// coverage as evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FieldFan {
    /// Entries that record any value for this field.
    pub population: u64,
    /// How many of those carry the cue's own value. Never more than
    /// `population`, and zero for a value the store has never seen.
    pub holding: u64,
}

impl FieldFan {
    /// What knowing this value is worth over this field's own population, in
    /// nats, or zero where it separates nobody or the sample is too small to
    /// believe.
    ///
    /// `ln(population / holding)`. Three populations sit outside the formula and
    /// all three are worth nothing, for one reason rather than three: a value
    /// that separates nobody tells nobody anything.
    ///
    /// - `holding == population`: every entry that records the field carries it.
    /// - `holding == 0`: no entry carries it, so no candidate can match on the
    ///   field and it discriminates among none of them.
    /// - `population < SITUATION_MIN_POPULATION`: the field is recorded on too
    ///   few entries for its fan to be a measurement.
    pub fn information(self) -> f64 {
        if self.population < SITUATION_MIN_POPULATION
            || self.holding == 0
            || self.holding >= self.population
        {
            return 0.0;
        }
        (self.population as f64 / self.holding as f64).ln().max(0.0)
    }
}

/// The present situation, read against one store.
///
/// Carries the cue itself and how much each of its values separates one entry of
/// this store from another, in nats. The store is the only thing that can say
/// that, so an adapter measures it and hands it in - the same split
/// [`RecallDispersion`] makes for distance, and for the same reason: a count
/// measured over the candidates a lookup returned describes the near tail rather
/// than the source.
///
/// [`RecallDispersion`]: crate::ports::recall::RecallDispersion
#[derive(Debug, Clone, PartialEq)]
pub struct SituationCue {
    situation: Situation,
    /// Self-information of each stated value, in nats. Never negative, and zero
    /// for a value that separates nobody.
    information: BTreeMap<SituationField, f64>,
}

impl SituationCue {
    /// Read `situation` against a store, or `None` where the store cannot grade
    /// it.
    ///
    /// `population` is how many entries of the source carry any situation record
    /// at all, and `fan` is how many of those carry the cue's own value for each
    /// field. Both are counted over the whole source rather than over the
    /// candidates one lookup returned.
    ///
    /// `None` - the quiet answer, which leaves every entry ranked the way it
    /// ranked before this term existed - for a population under
    /// [`SITUATION_MIN_POPULATION`], for an empty situation, and for a count
    /// that is not a number.
    pub fn measured(
        situation: Situation,
        fans: &BTreeMap<SituationField, FieldFan>,
    ) -> Option<Self> {
        if situation.is_empty() {
            return None;
        }
        let measurable = situation.iter().any(|(field, _)| {
            fans.get(&field)
                .is_some_and(|fan| fan.population >= SITUATION_MIN_POPULATION)
        });
        if !measurable {
            return None;
        }
        let information: BTreeMap<SituationField, f64> = situation
            .iter()
            .map(|(field, _)| {
                let fan = fans.get(&field).copied().unwrap_or_default();
                (field, fan.information())
            })
            .collect();
        Some(Self {
            situation,
            information,
        })
    }

    /// The present situation this cue reads.
    pub fn situation(&self) -> &Situation {
        &self.situation
    }

    /// What knowing this cue's value for `field` is worth over this store, in
    /// nats. Zero for a field the cue does not state.
    pub fn information(&self, field: SituationField) -> f64 {
        self.information.get(&field).copied().unwrap_or(0.0)
    }

    /// Whether this cue can separate anything at all.
    ///
    /// True when every value it states is one the whole store shares or one no
    /// entry holds. A cue like that scores every entry zero, which is the
    /// honest answer rather than a broken one.
    pub fn is_empty(&self) -> bool {
        self.information.values().all(|nats| *nats <= 0.0)
    }

    /// How much of what this cue could have told us about `record`'s entry it
    /// actually told us, in `[0, 1]`.
    ///
    /// The ratio of the information the cue carries on the fields the entry
    /// could have matched, to the information it carries on the fields it did.
    /// Three rules, and each of them is one line of the sum:
    ///
    /// - **A field the entry has never been seen with is skipped entirely.** It
    ///   is in neither half, so it neither matches nor penalises. That is the
    ///   deliberate choice, and it has a price worth stating: an entry that
    ///   knows one thing about itself and gets that one thing right reaches full
    ///   coverage on less evidence than an entry that knows four. The
    ///   alternative - dividing by the whole cue - scores "we do not know" the
    ///   same as "we know, and it was somewhere else", and conflating an unknown
    ///   with a mismatch is the worse error for a store whose older half was
    ///   written before any of this was recorded.
    /// - **A field the entry knows and disagrees on is in the denominator
    ///   only.** It forfeits the lift rather than subtracting from the score, so
    ///   an entry can never end below what its own distance and history earned
    ///   it.
    /// - **A field neither side can separate on is worth zero to both halves,**
    ///   because its information is zero. The fan does that, not a special case.
    ///
    /// Zero when nothing is comparable, which covers the entry with no record,
    /// the store below its population floor, and the deployment with nothing
    /// connected.
    pub fn coverage(&self, record: &SituationRecord) -> f64 {
        let mut matched = 0.0;
        let mut comparable = 0.0;
        for (field, value) in self.situation.iter() {
            if !record.knows(field) {
                continue;
            }
            let information = self.information(field);
            comparable += information;
            if record.holds(field, value) {
                matched += information;
            }
        }
        if comparable <= 0.0 || !comparable.is_finite() {
            return 0.0;
        }
        (matched / comparable).clamp(0.0, 1.0)
    }
}

/// The local wall clock at `now` in `zone`, or `None` for a zone name no
/// database entry claims.
fn local_time(now: DateTime<Utc>, zone: &str) -> Option<chrono::NaiveDateTime> {
    let zone: chrono_tz::Tz = zone.trim().parse().ok()?;
    Some(now.with_timezone(&zone).naive_local())
}

/// The weekday's English name, lowercase, so a stored value does not depend on
/// a locale the reader does not share.
fn weekday_name(local: chrono::NaiveDateTime) -> &'static str {
    match local.weekday() {
        chrono::Weekday::Mon => "monday",
        chrono::Weekday::Tue => "tuesday",
        chrono::Weekday::Wed => "wednesday",
        chrono::Weekday::Thu => "thursday",
        chrono::Weekday::Fri => "friday",
        chrono::Weekday::Sat => "saturday",
        chrono::Weekday::Sun => "sunday",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn now() -> DateTime<Utc> {
        // A Thursday, 16:30 UTC.
        DateTime::parse_from_rfc3339("2026-08-06T16:30:00Z")
            .expect("a fixed clock parses")
            .with_timezone(&Utc)
    }

    /// A store in which every named field is recorded on a hundred entries and
    /// every named value is held by a quarter of them, so each field carries the
    /// same information and the arithmetic below is legible.
    fn even_fan(fields: &[SituationField]) -> BTreeMap<SituationField, FieldFan> {
        fields
            .iter()
            .map(|field| {
                (
                    *field,
                    FieldFan {
                        population: 100,
                        holding: 25,
                    },
                )
            })
            .collect()
    }

    /// Acceptance (#1125): a situation record is written with every observation,
    /// and every field of it is optional.
    ///
    /// Three observations of the same instant, differing only in what the
    /// deployment has connected. Each produces a record; none of them fails; and
    /// what each carries is exactly what it could read.
    #[test]
    fn a_situation_record_is_written_with_every_observation_and_every_field_is_optional() {
        let now = now();

        let everything = Situation::observe(
            now,
            &SituationSources {
                host: Some("workshop"),
                timezone: Some("Europe/London"),
            },
        );
        assert_eq!(everything.get(SituationField::Host), Some("workshop"));
        assert_eq!(
            everything.get(SituationField::TimeOfDay),
            Some("evening"),
            "16:30 UTC is 17:30 in London, which is the evening"
        );
        assert_eq!(everything.get(SituationField::Weekday), Some("thursday"));

        // A client that reported a host and no zone. The clock fields are gated
        // on the zone, so they are absent rather than guessed.
        let no_zone = Situation::observe(
            now,
            &SituationSources {
                host: Some("workshop"),
                ..SituationSources::default()
            },
        );
        assert_eq!(no_zone.get(SituationField::Host), Some("workshop"));
        assert_eq!(no_zone.get(SituationField::TimeOfDay), None);
        assert_eq!(no_zone.get(SituationField::Weekday), None);

        // A client that reported a zone and no host.
        let no_host = Situation::observe(
            now,
            &SituationSources {
                timezone: Some("Europe/London"),
                ..SituationSources::default()
            },
        );
        assert_eq!(no_host.get(SituationField::Host), None);
        assert_eq!(no_host.get(SituationField::TimeOfDay), Some("evening"));

        // Nothing connected at all: an empty record, and no error.
        assert!(Situation::observe(now, &SituationSources::default()).is_empty());
    }

    /// Acceptance (#1125): a missing field neither matches nor penalises.
    ///
    /// Two entries against the same two-field cue. One has been seen in the
    /// cue's host and has never been seen with a time of day at all; the other
    /// has been seen in the cue's host and at a different time. The first must
    /// not be penalised for what it does not know, and the second must not be
    /// credited for what it knows and got wrong.
    #[test]
    fn a_missing_field_neither_matches_nor_penalises() {
        let fans = even_fan(&[SituationField::Host, SituationField::TimeOfDay]);
        let cue = SituationCue::measured(
            Situation::new()
                .with(SituationField::Host, "workshop")
                .with(SituationField::TimeOfDay, "morning"),
            &fans,
        )
        .expect("a hundred entries is a gradeable store");

        let silent_about_time = SituationRecord::new().with(SituationField::Host, "workshop");
        let seen_at_another_time = SituationRecord::new()
            .with(SituationField::Host, "workshop")
            .with(SituationField::TimeOfDay, "night");

        assert_eq!(
            cue.coverage(&silent_about_time),
            1.0,
            "a field the entry has never been seen with must not be held against it"
        );
        assert!(
            cue.coverage(&seen_at_another_time) < cue.coverage(&silent_about_time),
            "an entry known to belong to another time must not score as well as one that \
             says nothing about time"
        );
        assert!(
            cue.coverage(&seen_at_another_time) > 0.0,
            "the host still matched, so the entry keeps what that is worth"
        );
    }

    /// Acceptance (#1125): an entry written in a situation is ranked above an
    /// equally similar entry written elsewhere, when the situation recurs.
    ///
    /// Stated here as the coverage the two earn, which is what the ranking then
    /// spends; `crate::domain::activation` holds the same claim as scores, and
    /// `crate::recall` holds it as the order of a rendered block.
    #[test]
    fn an_entry_written_in_the_recurring_situation_outranks_one_written_elsewhere() {
        let fans = even_fan(&[SituationField::Host, SituationField::Weekday]);
        let thursday_at_the_workshop = Situation::new()
            .with(SituationField::Host, "workshop")
            .with(SituationField::Weekday, "thursday");
        let cue = SituationCue::measured(thursday_at_the_workshop.clone(), &fans)
            .expect("a hundred entries is a gradeable store");

        let written_here = SituationRecord::from_pairs([
            (SituationField::Host, "workshop"),
            (SituationField::Weekday, "thursday"),
        ]);
        let written_elsewhere = SituationRecord::from_pairs([
            (SituationField::Host, "the-road"),
            (SituationField::Weekday, "sunday"),
        ]);

        assert_eq!(cue.coverage(&written_here), 1.0);
        assert_eq!(cue.coverage(&written_elsewhere), 0.0);
    }

    /// Acceptance (#1125): the term's size does not grow with how many situation
    /// fields a deployment has connected.
    ///
    /// One field connected and all three connected, each matched in full. The
    /// coverage is the same number, because it is a ratio of the same quantity
    /// over itself - not a sum that a fourth source would enlarge.
    #[test]
    fn coverage_does_not_grow_with_the_number_of_fields_connected() {
        let one_field = even_fan(&[SituationField::Host]);
        let every_field = even_fan(&SituationField::ALL);

        let narrow = SituationCue::measured(
            Situation::new().with(SituationField::Host, "workshop"),
            &one_field,
        )
        .expect("gradeable");
        let wide = SituationCue::measured(
            Situation::new()
                .with(SituationField::Host, "workshop")
                .with(SituationField::TimeOfDay, "morning")
                .with(SituationField::Weekday, "thursday"),
            &every_field,
        )
        .expect("gradeable");

        let matches_what_it_can = SituationRecord::from_pairs([
            (SituationField::Host, "workshop"),
            (SituationField::TimeOfDay, "morning"),
            (SituationField::Weekday, "thursday"),
        ]);

        assert_eq!(
            narrow.coverage(&matches_what_it_can),
            wide.coverage(&matches_what_it_can),
            "a deployment that connects three sources must not lift its entries three times \
             as far as one that connects a single source"
        );
    }

    /// Acceptance (#1125): with no situation sources connected there is no cue,
    /// so nothing this module produces can move a ranking.
    #[test]
    fn with_no_situation_sources_connected_there_is_no_cue() {
        let fans = even_fan(&SituationField::ALL);
        assert_eq!(SituationCue::measured(Situation::new(), &fans), None);
        assert_eq!(
            SituationCue::measured(
                Situation::observe(now(), &SituationSources::default()),
                &fans
            ),
            None
        );
    }

    /// The fan effect, which is why the weight is information and not a count.
    ///
    /// A store with one host. Every entry carries it, so the cue value separates
    /// nobody and is worth nothing - and the term stays silent instead of
    /// lifting every entry that happens to have a record over every entry
    /// written before records existed.
    #[test]
    fn a_value_the_whole_store_shares_is_worth_nothing() {
        let fans = BTreeMap::from([(
            SituationField::Host,
            FieldFan {
                population: 100,
                holding: 100,
            },
        )]);
        let cue = SituationCue::measured(
            Situation::new().with(SituationField::Host, "the-only-host"),
            &fans,
        )
        .expect("gradeable");

        assert!(cue.is_empty(), "a cue that separates nobody is no cue");
        assert_eq!(
            cue.coverage(&SituationRecord::new().with(SituationField::Host, "the-only-host")),
            0.0
        );
    }

    /// A field only some entries record is measured against those entries, not
    /// against the whole store.
    ///
    /// The store has one host, recorded on a third of its entries because the
    /// rest were written by a client that reported no hostname. Among the
    /// entries that have a host, that host separates nobody, so it must be
    /// worth nothing - and it is only worth nothing if the denominator is the
    /// entries that record the field. Divided by the store-wide count it would
    /// come out informative, and an entry would then be lifted for matching a
    /// value no entry could fail to match, which is the exact failure the fan
    /// weighting exists to prevent.
    #[test]
    fn a_field_only_some_entries_record_is_measured_against_those_entries() {
        let one_host_on_a_third = BTreeMap::from([(
            SituationField::Host,
            FieldFan {
                population: 40,
                holding: 40,
            },
        )]);
        let cue = SituationCue::measured(
            Situation::new().with(SituationField::Host, "the-only-host"),
            &one_host_on_a_third,
        )
        .expect("forty entries is a gradeable field");

        assert_eq!(
            cue.information(SituationField::Host),
            0.0,
            "the only host among the entries that record one separates nobody"
        );
        assert_eq!(
            cue.coverage(&SituationRecord::new().with(SituationField::Host, "the-only-host")),
            0.0
        );
    }

    /// A field the store has too few records of is weighted at zero, while the
    /// fields beside it keep theirs.
    ///
    /// Two clients with different reach leave a store where the weekday is on
    /// every entry and the host is on three. A host measured from three samples
    /// is noise, and noise weighted at `ln(3)` would swamp a weekday measured
    /// over hundreds.
    #[test]
    fn a_field_measured_over_too_few_entries_is_weighted_at_zero() {
        let uneven = BTreeMap::from([
            (
                SituationField::Host,
                FieldFan {
                    population: 3,
                    holding: 1,
                },
            ),
            (
                SituationField::Weekday,
                FieldFan {
                    population: 700,
                    holding: 100,
                },
            ),
        ]);
        let cue = SituationCue::measured(
            Situation::new()
                .with(SituationField::Host, "the-boat")
                .with(SituationField::Weekday, "thursday"),
            &uneven,
        )
        .expect("the weekday is measurable, so the cue is");

        assert_eq!(cue.information(SituationField::Host), 0.0);
        assert!(cue.information(SituationField::Weekday) > 0.0);

        // So the host neither lifts nor suppresses: an entry that matches only
        // the weekday scores the same as one that matches both.
        let weekday_only = SituationRecord::from_pairs([
            (SituationField::Host, "elsewhere"),
            (SituationField::Weekday, "thursday"),
        ]);
        let both = SituationRecord::from_pairs([
            (SituationField::Host, "the-boat"),
            (SituationField::Weekday, "thursday"),
        ]);
        assert_eq!(cue.coverage(&weekday_only), cue.coverage(&both));
        assert_eq!(cue.coverage(&both), 1.0);
    }

    /// The other end of the same rule: a value no entry has been seen with
    /// separates nobody either, so it neither lifts nor suppresses.
    ///
    /// Without this, a cue value the store has never met would be maximally
    /// informative and would sit in every comparable entry's denominator,
    /// silencing the fields that did match.
    #[test]
    fn a_value_no_entry_holds_is_worth_nothing_rather_than_everything() {
        let fans = BTreeMap::from([
            (
                SituationField::Host,
                FieldFan {
                    population: 100,
                    holding: 25,
                },
            ),
            (
                SituationField::TimeOfDay,
                FieldFan {
                    population: 100,
                    holding: 0,
                },
            ),
        ]);
        let cue = SituationCue::measured(
            Situation::new()
                .with(SituationField::Host, "workshop")
                .with(SituationField::TimeOfDay, "night"),
            &fans,
        )
        .expect("gradeable");

        assert_eq!(cue.information(SituationField::TimeOfDay), 0.0);

        let matched_the_host = SituationRecord::from_pairs([
            (SituationField::Host, "workshop"),
            (SituationField::TimeOfDay, "morning"),
        ]);
        assert_eq!(
            cue.coverage(&matched_the_host),
            1.0,
            "a time of day nothing in the store holds must not silence the host that matched"
        );
    }

    /// A rarer value is worth more than a common one, which is what makes the
    /// weight a measurement rather than a label.
    #[test]
    fn a_rarer_value_carries_more_information_than_a_common_one() {
        let fans = BTreeMap::from([
            (
                SituationField::Host,
                FieldFan {
                    population: 1_000,
                    holding: 5,
                },
            ),
            (
                SituationField::Weekday,
                FieldFan {
                    population: 1_000,
                    holding: 500,
                },
            ),
        ]);
        let cue = SituationCue::measured(
            Situation::new()
                .with(SituationField::Host, "the-boat")
                .with(SituationField::Weekday, "monday"),
            &fans,
        )
        .expect("gradeable");

        assert!(cue.information(SituationField::Host) > cue.information(SituationField::Weekday));

        // And it shows in the ratio: matching only the rare value beats matching
        // only the common one.
        let rare_only = SituationRecord::from_pairs([
            (SituationField::Host, "the-boat"),
            (SituationField::Weekday, "friday"),
        ]);
        let common_only = SituationRecord::from_pairs([
            (SituationField::Host, "the-office"),
            (SituationField::Weekday, "monday"),
        ]);
        assert!(cue.coverage(&rare_only) > cue.coverage(&common_only));
    }

    /// A store too small to measure a fan over produces no cue, so a young
    /// deployment ranks exactly as it did before this term existed.
    #[test]
    fn a_store_below_the_population_floor_produces_no_cue() {
        let situation = Situation::new().with(SituationField::Host, "workshop");
        let below = BTreeMap::from([(
            SituationField::Host,
            FieldFan {
                population: SITUATION_MIN_POPULATION - 1,
                holding: 1,
            },
        )]);
        let at_the_floor = BTreeMap::from([(
            SituationField::Host,
            FieldFan {
                population: SITUATION_MIN_POPULATION,
                holding: 1,
            },
        )]);

        assert_eq!(SituationCue::measured(situation.clone(), &below), None);
        assert!(SituationCue::measured(situation, &at_the_floor).is_some());
    }

    /// An entry with no record scores zero, which is the same as not being
    /// scored at all.
    #[test]
    fn an_entry_with_no_situation_record_scores_zero() {
        let fans = even_fan(&SituationField::ALL);
        let cue = SituationCue::measured(
            Situation::new().with(SituationField::Host, "workshop"),
            &fans,
        )
        .expect("gradeable");

        assert_eq!(cue.coverage(&SituationRecord::new()), 0.0);
    }

    /// Accumulating a value the record already holds changes nothing, which is
    /// what closes the retrieve-record-retrieve loop after one step.
    #[test]
    fn recording_a_situation_the_entry_already_holds_changes_no_score() {
        let fans = even_fan(&[SituationField::Host]);
        let cue = SituationCue::measured(
            Situation::new().with(SituationField::Host, "workshop"),
            &fans,
        )
        .expect("gradeable");

        let once = SituationRecord::new().with(SituationField::Host, "workshop");
        let again = once.clone().with(SituationField::Host, "workshop");

        assert_eq!(once, again, "presence is the record, not how often");
        assert_eq!(cue.coverage(&once), cue.coverage(&again));
    }

    /// Coverage is a ratio, so it never leaves `[0, 1]` however the counts are
    /// shaped - which is what makes the term's bound a property of the function
    /// rather than a clamp at the call site.
    #[test]
    fn coverage_stays_inside_zero_and_one_over_every_shape_of_store() {
        for population in [SITUATION_MIN_POPULATION, 100, 10_000] {
            for holding in [0, 1, 7, population / 2, population] {
                let fan = FieldFan {
                    population,
                    holding,
                };
                let fans = BTreeMap::from([
                    (SituationField::Host, fan),
                    (SituationField::TimeOfDay, fan),
                    (SituationField::Weekday, fan),
                ]);
                let Some(cue) = SituationCue::measured(
                    Situation::new()
                        .with(SituationField::Host, "workshop")
                        .with(SituationField::TimeOfDay, "morning")
                        .with(SituationField::Weekday, "thursday"),
                    &fans,
                ) else {
                    continue;
                };
                for record in [
                    SituationRecord::new(),
                    SituationRecord::from_pairs([(SituationField::Host, "workshop")]),
                    SituationRecord::from_pairs([(SituationField::Host, "elsewhere")]),
                    SituationRecord::from_pairs([
                        (SituationField::Host, "workshop"),
                        (SituationField::TimeOfDay, "morning"),
                        (SituationField::Weekday, "thursday"),
                    ]),
                ] {
                    let coverage = cue.coverage(&record);
                    assert!(
                        (0.0..=1.0).contains(&coverage),
                        "population {population}, holding {holding} produced a coverage of \
                         {coverage}"
                    );
                }
            }
        }
    }

    /// The clock fields are read in the person's own zone, not the daemon's.
    ///
    /// One instant, two people. The same UTC timestamp is a Thursday evening in
    /// London and a Thursday morning in Los Angeles, and an entry written by one
    /// must not be treated as written in the other's part of the day.
    #[test]
    fn the_time_of_day_is_read_in_the_persons_own_zone() {
        let now = now();
        let london = Situation::observe(
            now,
            &SituationSources {
                timezone: Some("Europe/London"),
                ..SituationSources::default()
            },
        );
        let los_angeles = Situation::observe(
            now,
            &SituationSources {
                timezone: Some("America/Los_Angeles"),
                ..SituationSources::default()
            },
        );

        assert_eq!(london.get(SituationField::TimeOfDay), Some("evening"));
        assert_eq!(los_angeles.get(SituationField::TimeOfDay), Some("morning"));
    }

    /// A zone name nothing recognises is an absent zone, not a wrong one.
    #[test]
    fn an_unparseable_timezone_leaves_the_clock_fields_absent() {
        for zone in ["", "   ", "Mars/Olympus_Mons", "GMT+0700"] {
            let observed = Situation::observe(
                now(),
                &SituationSources {
                    host: Some("workshop"),
                    timezone: Some(zone),
                },
            );
            assert_eq!(observed.get(SituationField::Host), Some("workshop"));
            assert_eq!(
                observed.get(SituationField::TimeOfDay),
                None,
                "{zone:?} names no zone, so the clock fields must stay absent"
            );
            assert_eq!(observed.get(SituationField::Weekday), None);
        }
    }

    /// Every part of the day is reachable, and the boundaries fall where the
    /// documentation says they do.
    #[test]
    fn every_hour_of_the_day_lands_in_a_named_part() {
        assert_eq!(TimeOfDay::at_hour(4), TimeOfDay::Night);
        assert_eq!(TimeOfDay::at_hour(5), TimeOfDay::Morning);
        assert_eq!(TimeOfDay::at_hour(11), TimeOfDay::Morning);
        assert_eq!(TimeOfDay::at_hour(12), TimeOfDay::Afternoon);
        assert_eq!(TimeOfDay::at_hour(16), TimeOfDay::Afternoon);
        assert_eq!(TimeOfDay::at_hour(17), TimeOfDay::Evening);
        assert_eq!(TimeOfDay::at_hour(21), TimeOfDay::Evening);
        assert_eq!(TimeOfDay::at_hour(22), TimeOfDay::Night);
        assert_eq!(TimeOfDay::at_hour(0), TimeOfDay::Night);
    }

    /// A situation value is bounded and cleaned before it is stored, because a
    /// client chooses it and the store makes it part of an index key.
    ///
    /// A btree tuple has a hard size limit, so an unbounded value would let a
    /// client pick one the database refuses - and the write carrying it would
    /// fail rather than the value merely being odd. A NUL byte is worse: a text
    /// column cannot hold one, so it raises on the wire.
    #[test]
    fn a_situation_value_is_bounded_and_cleaned_before_it_is_stored() {
        let far_too_long = "h".repeat(MAX_SITUATION_VALUE_CHARS * 40);
        let situation = Situation::new().with(SituationField::Host, &far_too_long);
        let stored = situation
            .get(SituationField::Host)
            .expect("an over-long value is cut, not dropped");
        assert_eq!(stored.chars().count(), MAX_SITUATION_VALUE_CHARS);

        let hostile = Situation::new().with(SituationField::Host, "work\0shop\nkb-forged");
        assert_eq!(
            hostile.get(SituationField::Host),
            Some("workshopkb-forged"),
            "a value a text column cannot hold, or that forges a line, is cleaned"
        );

        // Nothing usable left is the same answer as a field with no source.
        for nothing in ["", "   ", "\0", "\n\t"] {
            assert!(
                Situation::new()
                    .with(SituationField::Host, nothing)
                    .is_empty()
            );
            assert!(
                SituationRecord::new()
                    .with(SituationField::Host, nothing)
                    .is_empty()
            );
        }
    }

    /// A host matches whatever case the client reported it in.
    ///
    /// One machine that answers `Workshop` to one client and `workshop` to
    /// another would otherwise hold two values, halve its own fan, and match
    /// neither prompt in full.
    #[test]
    fn a_host_is_recorded_in_one_case_whatever_the_client_reported() {
        let sources = |host| SituationSources {
            host: Some(host),
            ..SituationSources::default()
        };
        assert_eq!(
            Situation::observe(now(), &sources("Workshop")),
            Situation::observe(now(), &sources("  workshop ")),
        );
        assert_eq!(
            Situation::observe(now(), &sources("WORKSHOP")).get(SituationField::Host),
            Some("workshop")
        );
    }

    /// A field name round-trips, and a name from a later version is skipped
    /// rather than refused.
    #[test]
    fn a_stored_field_name_round_trips_and_an_unknown_one_is_skipped() {
        for field in SituationField::ALL {
            assert_eq!(SituationField::parse(field.as_str()), Some(field));
        }
        assert_eq!(SituationField::parse("calendar_event"), None);
    }
}
