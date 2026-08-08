//! Negative memory: an action that went badly, and the context it went badly
//! in (#1126).
//!
//! A fact and a burn are not the same kind of memory, and giving the second one
//! the first one's policy gets it wrong in every way that matters. This module
//! is the whole rule: what a burn is, when it fires, when it goes quiet, and
//! what it takes to widen one.
//!
//! ## Strength and scope are two axes, and they move in opposite directions
//!
//! This is the idea the rest of the module falls out of.
//!
//! - **Strength is full on the first bad outcome.** A fact earns strength by
//!   reinforcement; a burn does not have that time. The second occurrence is
//!   the one the memory exists to prevent, so the first one writes at
//!   [`FULL_STRENGTH`] and nothing raises it further. Strength only ever falls,
//!   by [`NegativeMemory::strength`], and a burn that falls under
//!   [`SILENCE_FLOOR`] stops firing.
//! - **Scope is as narrow as the evidence allows.** Every facet the failure was
//!   observed in is required, so a fresh burn fires only on a repeat of exactly
//!   what went wrong. It widens by [`Scope::broadened_against`], and only a
//!   second occurrence can widen it.
//!
//! Over-generalization is the failure mode this shape exists to prevent. An
//! assistant that turns "this failed once" into "never do this" becomes
//! uselessly cautious, and the caution is invisible: it presents as reticence
//! rather than as an error. A narrow burn that rarely fires is the safe
//! mistake; a wide one that fires everywhere is not.
//!
//! ## What a burn is keyed on
//!
//! Two things, and only the second one ever moves.
//!
//! - **The act**: the tool's name and [`PendingAction::fingerprint`], a digest
//!   over the call's arguments exactly as they were made. This is the burn's
//!   identity, and it never widens. A tool call is the one thing in a turn with
//!   an identity that can be matched exactly, and exact matching is what holds
//!   a burn to the act it was actually about.
//! - **The circumstance**: the situation facets of
//!   [`crate::domain::situation`] - where the act was taken, and when. These
//!   are what a second occurrence drops, once it shows a value was not the
//!   cause.
//!
//! That is the whole of "scope tightly, broaden only on re-confirmation": the
//! act is fixed evidence, the circumstance is provisional, and only an observed
//! disagreement makes the burn wider.
//!
//! **Firing and confirming ask the same question.** A burn fires on the act it
//! was recorded against and on no other, so a call whose arguments differ in
//! any way - one value changed, one argument added, one left out - is a
//! different act and a different lesson. That is stricter than it has to be,
//! deliberately: a burn that fails to fire costs one repeat of a mistake, and a
//! burn that fires on the wrong act costs the assistant's usefulness in a way
//! nobody can see.
//!
//! ## Why the digest reads the whole argument and the record does not
//!
//! [`Scope`] stores only argument values short enough to hold whole, because a
//! stored value is there to be read back and shown. The digest has no such
//! limit: it hashes every argument at full length. Without that, two failures
//! of one tool differing only past the storable length would share a
//! fingerprint and fold into one lesson, which is the same over-generalization
//! by another route - and it would land hardest on `command`, `query` and
//! `url`, the arguments that *are* the act.
//!
//! ## Extinction is an overlay
//!
//! A burn that stops applying is not deleted. The correction is written as its
//! own [`NegativeMemoryKind::Correction`] row, the burn's `superseded_by` names
//! it, and the burn stays readable. "This action went badly, and later it
//! stopped going badly" is knowledge; deleting it lets the same lesson be
//! learned again from nothing.
//!
//! ## Three negatives, and this is only one of them
//!
//! | | says | acts on | retrieved |
//! | --- | --- | --- | --- |
//! | negative mark ([`KnowledgeMark`]) | this entry was retrieved and was useless | the store, as prune evidence | never |
//! | a refuted claim | this claim is untrue | content: a negative fact | when the query is *about* the subject |
//! | negative memory (here) | this action in this context went badly | content: a negative procedure | at the decision point, before acting |
//!
//! Nothing here reads or writes a [`KnowledgeMark`], and marking an entry
//! writes nothing here. The two answer different questions about different
//! objects, and a shared implementation would have to pick one of them to be
//! wrong about.
//!
//! [`KnowledgeMark`]: crate::domain::knowledge_use::KnowledgeMark
//!
//! ## The strength scale is not the activation scale
//!
//! [`crate::domain::activation`] states its terms in a source's own median
//! absolute deviations, because it ranks candidates against each other. Nothing
//! here is ranked against anything: a burn either interrupts an action or it
//! does not. So strength is a plain fraction of full, it never enters `A_i`,
//! and no weight fitted there means anything here.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::domain::situation::{MAX_SITUATION_VALUE_CHARS, Situation, SituationField};

/// The strength a burn is written at, and the most it ever holds.
///
/// One trial is enough. Nothing raises a burn above this, and re-confirmation
/// restores it here rather than pushing it past - there is no "more certain
/// than certain", and a counter that could grow would make an old lesson
/// louder than a new one for no reason but its age.
pub const FULL_STRENGTH: f64 = 1.0;

/// Days for an unconfirmed burn's strength to halve.
///
/// Two weeks, chosen against how quickly the things a burn is about change. A
/// broken script gets fixed, an interface moves, a permission is granted; none
/// of that announces itself, so an unrepeated burn has to become wrong by
/// default rather than stay right until something proves it wrong. Two weeks is
/// long enough that a fortnightly task meets its own lesson, and short enough
/// that a one-off failure is not still interrupting work next quarter.
pub const HALF_LIFE_DAYS: f64 = 14.0;

/// Strength under which a burn stops firing.
///
/// A quarter of full, which at [`HALF_LIFE_DAYS`] is four weeks after the last
/// confirmation. The burn is not gone at that point - it stays readable, and a
/// fresh occurrence confirms it back to full rather than starting a new
/// lesson - it just stops interrupting.
pub const SILENCE_FLOOR: f64 = 0.25;

/// How far ahead of the reader's clock a confirmation stamp may sit and still
/// be believed, in hours.
///
/// The stamp is written by the database's `NOW()` and read against the daemon's
/// clock, so a little skew between the two is ordinary and means nothing. An
/// hour is far past ordinary.
///
/// A stamp further ahead than this is not a fresher lesson, it is a broken
/// clock, and the arithmetic would otherwise read it as full strength for as
/// long as the skew lasts - a burn that can be neither silenced nor reaped,
/// firing the whole time. That is the fires-when-it-should-not direction, so
/// such a stamp is disbelieved rather than trusted: the burn scores zero, goes
/// quiet, and becomes reapable. See [`NegativeMemory::strength`].
///
/// Whole hours, and the type says so: the store binds this straight into an
/// interval, so a fractional value would be rounded there and not here, and the
/// two would disagree about which rows are dead.
pub const FUTURE_STAMP_TOLERANCE_HOURS: u32 = 1;

/// Days after which a burn nothing has confirmed is dropped from the store.
///
/// Four half-lives, so a sixteenth of full strength and twice as long as
/// silence. Deleting it loses nothing a reader would act on and bounds the
/// table without a sweep: the writer reaps on its own path, the way the
/// situation record's writer bounds itself.
///
/// Whole days, and the type says so, for the reason
/// [`FUTURE_STAMP_TOLERANCE_HOURS`] gives.
pub const FORGET_DAYS: u32 = 56;

/// Most argument facets one burn records.
///
/// A bound on what is *stored and shown*, not on what is matched: the identity
/// is [`PendingAction::fingerprint`], which reads every argument whatever this
/// says. So trimming here cannot widen a burn - it can only shorten the line
/// that explains one. Arguments past the cap are dropped in name order, which
/// is stable across calls.
pub const MAX_ACTION_FACETS: usize = 12;

/// Longest argument value a burn records.
///
/// The same bound the situation record uses for its own values. A longer value
/// is not truncated to fit - a truncated value read back would misdescribe the
/// call - it is simply not recorded. Nothing is lost from the match by that:
/// [`PendingAction::fingerprint`] has already read the value at full length.
pub const MAX_FACET_VALUE_CHARS: usize = MAX_SITUATION_VALUE_CHARS;

/// Longest outcome text a burn carries.
pub const MAX_OUTCOME_CHARS: usize = 400;

/// Most burns the warning names at one decision point.
///
/// A wall of past failures is not a warning, it is noise, and the model has to
/// read all of it before it can act.
pub const MAX_WARNED_BURNS: usize = 3;

/// Most live burns a read returns for one user.
///
/// The whole live set is read once per turn and matched in memory, so this is
/// what bounds that read.
///
/// What happens past it, rather than a prediction that nobody gets there: the
/// read is ordered by last confirmation, so a user holding more keeps the most
/// recently confirmed and the rest stop firing. That is the safe direction - a
/// lesson that goes quiet costs one repeat of a mistake, where the alternative
/// bound is an unbounded read on every turn.
pub const MAX_LIVE_BURNS: usize = 200;

/// One dimension a burn is scoped by.
///
/// The two kinds are not interchangeable - see the module header. An argument
/// is identity and never widens; a situation value is circumstance and is what
/// a second occurrence drops.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Facet {
    /// One of the action's own arguments, by name.
    Argument(String),
    /// Where and when the action was taken (#1125).
    Situation(SituationField),
}

impl Facet {
    /// The stored discriminator: `argument` or `situation`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Argument(_) => "argument",
            Self::Situation(_) => "situation",
        }
    }

    /// The stored name within the kind.
    pub fn name(&self) -> &str {
        match self {
            Self::Argument(name) => name.as_str(),
            Self::Situation(field) => field.as_str(),
        }
    }

    /// Rebuild a facet from its stored `(kind, name)` pair.
    ///
    /// `None` for a kind this build does not know, and for a situation field
    /// name it cannot parse - an unknown dimension is one that cannot be
    /// matched, and a burn holding one must not fire as though it had matched.
    pub fn from_stored(kind: &str, name: &str) -> Option<Self> {
        match kind {
            "argument" => Some(Self::Argument(name.to_string())),
            "situation" => SituationField::parse(name).map(Self::Situation),
            _ => None,
        }
    }

    /// Whether this facet is part of the action's identity.
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Argument(_))
    }
}

/// The facets an action was taken with, or must be taken with for a burn to
/// fire.
///
/// One value per facet, unlike the situation *record* of
/// [`crate::domain::situation`], which accumulates every value an entry has
/// been seen in. The difference is the point: a record is a history, and a
/// scope is a condition.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scope(BTreeMap<Facet, String>);

impl Scope {
    /// An empty scope, which every action matches.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Add one facet, replacing any value already held for it.
    #[must_use]
    pub fn with(mut self, facet: Facet, value: impl Into<String>) -> Self {
        self.0.insert(facet, value.into());
        self
    }

    /// Rebuild a scope from stored `(kind, name, value)` triples.
    ///
    /// `None` when any row names a facet this build cannot resolve. Dropping
    /// the row instead would be the dangerous answer, not the safe one: the
    /// burn would lose a requirement and fire on acts it had never been seen
    /// with. A memory this build cannot read whole is a memory it must not act
    /// on at all.
    pub fn from_stored<I, K, N, V>(rows: I) -> Option<Self>
    where
        I: IntoIterator<Item = (K, N, V)>,
        K: AsRef<str>,
        N: AsRef<str>,
        V: Into<String>,
    {
        rows.into_iter()
            .map(|(kind, name, value)| {
                Facet::from_stored(kind.as_ref(), name.as_ref()).map(|f| (f, value.into()))
            })
            .collect::<Option<BTreeMap<_, _>>>()
            .map(Self)
    }

    /// Every facet and its required value, in a stable order.
    pub fn iter(&self) -> impl Iterator<Item = (&Facet, &str)> {
        self.0.iter().map(|(f, v)| (f, v.as_str()))
    }

    /// The value required for `facet`, if any.
    pub fn get(&self, facet: &Facet) -> Option<&str> {
        self.0.get(facet).map(String::as_str)
    }

    /// How many facets this scope requires.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this scope requires nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether `present` satisfies every facet this scope requires.
    ///
    /// Only ever asked of two scopes whose acts already match, so in practice
    /// this decides the circumstance: the argument facets on both sides are
    /// identical whenever the fingerprints are. Checking them anyway costs
    /// nothing and keeps the answer right if a caller ever asks without having
    /// compared the act first.
    pub fn matches(&self, present: &Scope) -> bool {
        self.0
            .iter()
            .all(|(facet, value)| present.get(facet) == Some(value.as_str()))
    }

    /// The same scope with nothing the model wrote in it: circumstance only.
    ///
    /// An argument value is chosen by the model, and a model that has just read
    /// a web page may be echoing it. The values are shown to a later turn at a
    /// decision point, so where the turn cannot vouch for them they are not
    /// recorded at all. Dropping them costs a warning some of what it can say
    /// and costs the match nothing: the act is the digest, which this does not
    /// touch.
    #[must_use]
    pub fn without_arguments(&self) -> Scope {
        Scope(
            self.0
                .iter()
                .filter(|(facet, _)| !facet.is_identity())
                .map(|(f, v)| (f.clone(), v.clone()))
                .collect(),
        )
    }

    /// This scope widened by a second occurrence.
    ///
    /// A situation facet the second occurrence *observed with a different
    /// value* is dropped: the failure happened without it, so it was not the
    /// cause.
    ///
    /// A facet the second occurrence says nothing about is kept. Absence is not
    /// disagreement, and treating it as disagreement would let one repeat from
    /// a headless path - or from a client that reports no hostname and no
    /// timezone, where the whole situation is empty - strip a burn's
    /// circumstance and set it firing everywhere. That is widening on the
    /// absence of evidence, which is the failure mode this module exists to
    /// prevent, arriving as its own remedy.
    ///
    /// Identity facets are kept whatever `observed` says. Confirmation only
    /// reaches here for two occurrences of the same act, so they are equal
    /// already; the guard is what makes that structural rather than incidental.
    ///
    /// This is the only thing in the module that makes a burn wider.
    #[must_use]
    pub fn broadened_against(&self, observed: &Scope) -> Scope {
        Scope(
            self.0
                .iter()
                .filter(|(facet, value)| {
                    facet.is_identity()
                        || observed
                            .get(facet)
                            .is_none_or(|seen| seen == value.as_str())
                })
                .map(|(f, v)| (f.clone(), v.clone()))
                .collect(),
        )
    }
}

/// A stable digest of a call's arguments, exactly as they were made: the burn's
/// handle.
///
/// Two failures share a lesson when their action and their fingerprint both
/// match, and nothing else makes them one. Three properties it has to have, and
/// each is a line below rather than a convention:
///
/// - **It reads every argument at full length**, so two calls differing only
///   past [`MAX_FACET_VALUE_CHARS`] are two lessons rather than one.
/// - **It does not depend on the order arguments arrive in**, because an object
///   has no order and two encoders of one call must agree.
/// - **It cannot confuse a name for a value, or one argument for two.** Every
///   string is written with its length in front of it and every composite
///   frames its own end, so `{"ab": "c"}` and `{"a": "bc"}` differ - and so do
///   two calls whose argument names carry whatever byte a separator-only
///   framing would have relied on being absent.
///
/// The situation is excluded on purpose: it is what widens, so it cannot also
/// be what identifies.
pub fn fingerprint(arguments: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    absorb(&mut hasher, arguments, MAX_ARGUMENT_DEPTH);
    format!("{:x}", hasher.finalize())[..32].to_string()
}

/// How far into a nested argument the digest reads.
///
/// Past this it writes one marker byte and stops, which folds two calls
/// differing only below the limit into one act - the identity widening, and
/// that is the one thing this module promises never happens. So the limit has
/// to sit above anything that can reach it, not merely above anything likely
/// to.
///
/// A tool call's arguments arrive as parsed JSON, and the parser refuses more
/// than 128 levels of nesting, so 128 is the deepest value that can ever be
/// handed to [`fingerprint`] on the dispatch path. This is twice that. What the
/// marker path is left bounding is a value built programmatically rather than
/// parsed: that has no depth limit of its own and would otherwise recurse until
/// the stack ran out.
///
/// `the_digest_reads_every_argument_a_parser_will_accept` holds the claim
/// against the parser in the tree rather than against the number written here,
/// so a parser that grows a deeper limit fails the gate instead of quietly
/// folding two acts into one.
pub const MAX_ARGUMENT_DEPTH: usize = 256;

/// Feed one JSON value into `hasher` in a form that depends on its content and
/// not on how it was written.
fn absorb(hasher: &mut Sha256, value: &serde_json::Value, depth: usize) {
    let Some(depth) = depth.checked_sub(1) else {
        hasher.update(*b"!");
        return;
    };
    // Every arm leads with a distinct tag byte, so the string "1" and the
    // number 1 cannot hash alike, and each composite frames its own end.
    match value {
        serde_json::Value::Null => hasher.update(*b"0"),
        serde_json::Value::Bool(b) => {
            hasher.update(*b"b");
            hasher.update([u8::from(*b)]);
        }
        serde_json::Value::Number(n) => {
            hasher.update(*b"n");
            absorb_text(hasher, &n.to_string());
        }
        serde_json::Value::String(text) => {
            hasher.update(*b"s");
            absorb_text(hasher, text);
        }
        serde_json::Value::Array(items) => {
            hasher.update(*b"[");
            for item in items {
                absorb(hasher, item, depth);
            }
            hasher.update(*b"]");
        }
        serde_json::Value::Object(map) => {
            hasher.update(*b"{");
            // Sorted, so the digest does not depend on whether the map behind
            // `serde_json::Value` preserves insertion order.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            for key in keys {
                absorb_text(hasher, key);
                absorb(hasher, &map[key], depth);
            }
            hasher.update(*b"}");
        }
    }
}

/// Feed one string in with its length in front of it.
///
/// Length-prefixed rather than separator-framed, because a separator only
/// frames what cannot contain it - and a JSON string, including an object's
/// key, can contain any byte at all. A length says where the string ends
/// whatever is in it, which is what makes the digest injective rather than
/// injective-in-practice.
fn absorb_text(hasher: &mut Sha256, text: &str) {
    hasher.update((text.len() as u64).to_be_bytes());
    hasher.update(text.as_bytes());
}

/// A tool call about to be made, reduced to what a burn is matched on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingAction {
    /// The tool's name.
    pub action: String,
    /// The digest of the call's arguments: what makes this act this act. See
    /// [`fingerprint`].
    pub fingerprint: String,
    /// What the burn records and shows: the arguments short enough to hold
    /// whole, and the situation the call is made in.
    pub scope: Scope,
}

impl PendingAction {
    /// Read a pending call and the present situation into a matchable act.
    ///
    /// Total: every call can be scoped, because the identity is a digest of the
    /// arguments and a digest has no shape it cannot read. What varies is how
    /// much of the call the burn can *show*, and that is [`Scope`]'s business
    /// rather than the match's.
    ///
    /// An argument becomes a facet when it is a scalar with no control
    /// characters and is no longer than [`MAX_FACET_VALUE_CHARS`], and at most
    /// [`MAX_ACTION_FACETS`] of them are kept, in name order. Every one of
    /// those limits shortens what a warning can say and none of them widens
    /// what it fires on.
    pub fn observe(
        action: impl Into<String>,
        arguments: &serde_json::Value,
        situation: &Situation,
    ) -> Self {
        // A provider that emits no arguments may send an empty object or a
        // literal null, and the two are the same call. Normalising here keeps
        // one act from splitting into two lessons on a detail of the wire.
        let arguments = match arguments {
            serde_json::Value::Null => &serde_json::Value::Object(serde_json::Map::new()),
            other => other,
        };
        let mut scope = Scope::new();
        if let serde_json::Value::Object(map) = arguments {
            let mut named: Vec<(&String, String)> = map
                .iter()
                .filter_map(|(name, value)| facet_value(value).map(|v| (name, v)))
                .filter(|(name, _)| storable_text(name))
                .collect();
            named.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (name, value) in named.into_iter().take(MAX_ACTION_FACETS) {
                scope = scope.with(Facet::Argument(name.clone()), value);
            }
        }
        for (field, value) in situation.iter() {
            scope = scope.with(Facet::Situation(field), value);
        }

        Self {
            action: action.into(),
            fingerprint: fingerprint(arguments),
            scope,
        }
    }
}

/// What one stored row says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeMemoryKind {
    /// An action that went badly. The only kind that fires.
    Burn,
    /// The correction written over a burn that stopped applying. Readable,
    /// never fired.
    Correction,
}

impl NegativeMemoryKind {
    /// The stored value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::Correction => "correction",
        }
    }

    /// Read a stored value back. `None` for a kind this build does not know.
    pub fn from_stored(value: &str) -> Option<Self> {
        match value {
            "burn" => Some(Self::Burn),
            "correction" => Some(Self::Correction),
            _ => None,
        }
    }
}

/// One thing the assistant learned by being burned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeMemory {
    /// The row's own id.
    pub id: String,
    /// The tool the memory is about.
    pub action: String,
    /// The digest of the arguments the act was made with. With `action`, the
    /// identity a call has to match exactly before this fires. See
    /// [`fingerprint`].
    pub fingerprint: String,
    /// Whether this row is the lesson or the correction over it.
    pub kind: NegativeMemoryKind,
    /// Everything that must be true before this fires.
    pub scope: Scope,
    /// What went wrong, in the words of whatever recorded it.
    pub outcome: String,
    /// How many times this lesson has been recorded. Widening evidence, not
    /// strength: nothing here raises [`Self::strength`].
    pub occurrences: u32,
    /// When the lesson was first recorded.
    pub written_at: DateTime<Utc>,
    /// When it was last recorded again. Decay runs from here.
    pub last_confirmed_at: DateTime<Utc>,
    /// The correction that extinguished this, when one has.
    pub superseded_by: Option<String>,
}

impl NegativeMemory {
    /// How much of full strength is left, as a fraction.
    ///
    /// [`FULL_STRENGTH`] at the moment of confirmation, halving every
    /// [`HALF_LIFE_DAYS`] after it. Never negative, never above full, and it
    /// carries no unit that anything else in the workspace reads - see the
    /// module header.
    pub fn strength(&self, now: DateTime<Utc>) -> f64 {
        let days = self.days_since_confirmed(now);
        // A stamp from beyond the reader's clock is a broken clock, not a
        // fresher lesson - see `FUTURE_STAMP_TOLERANCE_HOURS`.
        if days < -future_tolerance_days() {
            return 0.0;
        }
        if days <= 0.0 {
            return FULL_STRENGTH;
        }
        FULL_STRENGTH * 0.5_f64.powf(days / HALF_LIFE_DAYS)
    }

    /// Days since the last confirmation, negative when the stamp sits ahead of
    /// `now`.
    ///
    /// One reading of the clock, so decay and reaping cannot come to different
    /// conclusions about how old a burn is.
    fn days_since_confirmed(&self, now: DateTime<Utc>) -> f64 {
        now.signed_duration_since(self.last_confirmed_at)
            .num_milliseconds() as f64
            / (1000.0 * 60.0 * 60.0 * 24.0)
    }

    /// Whether this has decayed past the point of interrupting anything.
    pub fn is_silent(&self, now: DateTime<Utc>) -> bool {
        self.strength(now) < SILENCE_FLOOR
    }

    /// Whether this should be dropped from the store.
    ///
    /// Two ways to qualify. The ordinary one is age: nothing has confirmed it
    /// for [`FORGET_DAYS`]. The other is a stamp from beyond the reader's
    /// clock, which scores zero and can never rise, so the row is dead weight
    /// the age rule alone would keep forever.
    ///
    /// The store's own reap states the same two rules in SQL. They have to
    /// agree - this is what a reader believes, and that is what actually
    /// happens.
    pub fn is_forgotten(&self, now: DateTime<Utc>) -> bool {
        let days = self.days_since_confirmed(now);
        days > f64::from(FORGET_DAYS) || days < -future_tolerance_days()
    }

    /// Whether this burn interrupts `pending`.
    ///
    /// Six conditions, and every one of them is a way the burn could be wrong
    /// about right now: it must be a lesson rather than a correction, it must
    /// not have been extinguished, it must be about this tool, it must be about
    /// this call rather than another call of the same tool, it must still be
    /// loud enough, and its circumstance must hold.
    pub fn fires(&self, pending: &PendingAction, now: DateTime<Utc>) -> bool {
        self.kind == NegativeMemoryKind::Burn
            && self.superseded_by.is_none()
            && self.action == pending.action
            && self.fingerprint == pending.fingerprint
            && !self.is_silent(now)
            && self.scope.matches(&pending.scope)
    }
}

/// Every burn that interrupts `pending`, strongest first.
///
/// All of them, not the few worth showing. The display cap belongs to
/// [`render_warning`], because the other caller of this function extinguishes
/// what a success disproved - and a lesson left standing because it fell off
/// the end of a warning would be a lesson nothing could ever correct.
pub fn burns_that_fire<'a>(
    memories: &'a [NegativeMemory],
    pending: &PendingAction,
    now: DateTime<Utc>,
) -> Vec<&'a NegativeMemory> {
    let mut fired: Vec<&NegativeMemory> =
        memories.iter().filter(|m| m.fires(pending, now)).collect();
    // Strongest first, then most-repeated, then by id so a tie is stable rather
    // than left to the order the store happened to answer in.
    fired.sort_by(|a, b| {
        b.strength(now)
            .total_cmp(&a.strength(now))
            .then(b.occurrences.cmp(&a.occurrences))
            .then(a.id.cmp(&b.id))
    });
    fired
}

/// What the model reads in place of the tool result, when a burn fires.
///
/// Deliberately the same shape of interruption as a surfaced procedure: a
/// candidate to check, not an instruction to obey. The two arrive at the same
/// moment and should read as one rule rather than as two unrelated warnings.
/// Returns `None` when nothing fired.
pub fn render_warning(fired: &[&NegativeMemory], now: DateTime<Utc>) -> Option<String> {
    if fired.is_empty() {
        return None;
    }
    let mut out = String::from(
        "This call has not run. The same call went badly before, and what follows is a \
         candidate warning, not a refusal.\n",
    );
    for burn in fired.iter().take(MAX_WARNED_BURNS) {
        let days = now
            .signed_duration_since(burn.last_confirmed_at)
            .num_days()
            .max(0);
        let when = match days {
            0 => "today".to_string(),
            1 => "yesterday".to_string(),
            n => format!("{n} days ago"),
        };
        let times = match burn.occurrences {
            0 | 1 => String::new(),
            n => format!(", {n} times"),
        };
        out.push_str(&format!("\n- Last {when}{times}: {}\n", burn.outcome));
        let arguments = describe(&burn.scope, false);
        if !arguments.is_empty() {
            out.push_str(&format!("  Called with: {arguments}\n"));
        }
        let circumstance = describe(&burn.scope, true);
        if !circumstance.is_empty() {
            out.push_str(&format!("  It went badly at: {circumstance}\n"));
        }
    }
    out.push_str(
        "\nDecide whether the cause still applies. If it does not - the fault is fixed, the \
         interface changed, or you mean something different this time - make the same call \
         again and it will run. If it does, take another way.",
    );
    Some(out)
}

/// One line naming half of what a scope holds, for the warning above.
///
/// The two halves are rendered apart because they mean different things, and
/// because an argument may be named `host`, which would otherwise read exactly
/// like the situation field of that name.
fn describe(scope: &Scope, situation: bool) -> String {
    scope
        .iter()
        .filter(|(facet, _)| facet.is_identity() != situation)
        .map(|(facet, value)| format!("{}={value}", facet.name()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The recorded form of an argument value, when it has one.
///
/// Scalars only, short enough to hold whole, and free of control characters -
/// which a database `text` column cannot always store and a warning cannot
/// legibly show. A value this refuses is still read at full length by
/// [`fingerprint`], so refusing one narrows what a warning can say and never
/// what it fires on.
fn facet_value(value: &serde_json::Value) -> Option<String> {
    let rendered = match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => return None,
    };
    (rendered.chars().count() <= MAX_FACET_VALUE_CHARS && storable_text(&rendered))
        .then_some(rendered)
}

/// Whether `text` can be held whole in a stored facet.
///
/// Postgres `text` cannot hold a NUL byte, and no control character belongs in
/// a value a person reads back off a warning.
fn storable_text(text: &str) -> bool {
    !text.is_empty() && !text.chars().any(char::is_control)
}

/// [`FUTURE_STAMP_TOLERANCE_HOURS`] as a fraction of a day, which is the unit
/// the decay arithmetic works in.
fn future_tolerance_days() -> f64 {
    f64::from(FUTURE_STAMP_TOLERANCE_HOURS) / 24.0
}

/// Cut `outcome` to [`MAX_OUTCOME_CHARS`] on a character boundary.
pub fn clamp_outcome(outcome: &str) -> String {
    let trimmed = outcome.trim();
    if trimmed.chars().count() <= MAX_OUTCOME_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(MAX_OUTCOME_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-07T09:00:00Z")
            .expect("the fixed clock is a valid timestamp")
            .with_timezone(&Utc)
    }

    /// The situation the tests below take their actions in: a Thursday morning
    /// at the workshop.
    fn here_and_now() -> Situation {
        Situation::new()
            .with(SituationField::Host, "workshop")
            .with(SituationField::TimeOfDay, "morning")
            .with(SituationField::Weekday, "thursday")
    }

    fn args(pairs: &[(&str, serde_json::Value)]) -> serde_json::Value {
        serde_json::Value::Object(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
        )
    }

    fn text(value: &str) -> serde_json::Value {
        serde_json::Value::String(value.to_string())
    }

    /// The call every test below is about: clearing a build directory that
    /// turned out to be a mount point.
    fn the_call() -> PendingAction {
        PendingAction::observe(
            "terminal_run",
            &args(&[("command", text("rm -rf build")), ("cwd", text("/srv/app"))]),
            &here_and_now(),
        )
    }

    /// The burn that call left behind, recorded `age` before the fixed clock.
    fn burn_aged(age: TimeDelta) -> NegativeMemory {
        let pending = the_call();
        NegativeMemory {
            id: "nm-1".to_string(),
            action: pending.action.clone(),
            fingerprint: pending.fingerprint.clone(),
            kind: NegativeMemoryKind::Burn,
            scope: pending.scope,
            outcome: "rm -rf failed: build is a mount point and the host lost its cache"
                .to_string(),
            occurrences: 1,
            written_at: now() - age,
            last_confirmed_at: now() - age,
            superseded_by: None,
        }
    }

    fn fresh_burn() -> NegativeMemory {
        burn_aged(TimeDelta::zero())
    }

    // --- Acceptance: one trial is enough -----------------------------------

    /// Acceptance (#1126): a burn holds full strength from the moment it is
    /// recorded, with no reinforcement, and it fires straight away.
    #[test]
    fn a_burn_holds_full_strength_and_fires_on_its_first_occurrence() {
        let burn = fresh_burn();
        assert_eq!(
            burn.occurrences, 1,
            "the first bad outcome is one occurrence"
        );
        assert!(
            (burn.strength(now()) - FULL_STRENGTH).abs() < f64::EPSILON,
            "a burn is written at full strength, not built up to it"
        );
        assert!(
            burn.fires(&the_call(), now()),
            "one bad outcome is enough to interrupt the same call again"
        );
    }

    /// Acceptance (#1126): re-confirming does not push a burn past full
    /// strength, so an old repeated lesson never outshouts a new one.
    #[test]
    fn confirming_a_burn_restores_full_strength_and_does_not_exceed_it() {
        let mut burn = burn_aged(TimeDelta::days(20));
        assert!(burn.strength(now()) < FULL_STRENGTH);
        burn.last_confirmed_at = now();
        burn.occurrences = 7;
        assert!(
            (burn.strength(now()) - FULL_STRENGTH).abs() < f64::EPSILON,
            "seven occurrences are worth exactly one: strength is not a counter"
        );
    }

    // --- Acceptance: context, action and outcome ---------------------------

    /// Acceptance (#1126): a burn records the action, the context it was taken
    /// in, and what went wrong - not a bare proposition.
    #[test]
    fn a_burn_records_the_action_the_context_and_the_outcome() {
        let burn = fresh_burn();
        assert_eq!(burn.action, "terminal_run", "the action is named");
        assert_eq!(
            burn.scope.get(&Facet::Argument("command".to_string())),
            Some("rm -rf build"),
            "the action's own arguments are part of the context"
        );
        assert_eq!(
            burn.scope.get(&Facet::Situation(SituationField::Host)),
            Some("workshop"),
            "the situation the action was taken in is part of the context"
        );
        assert!(
            burn.outcome.contains("mount point"),
            "the outcome says what went wrong"
        );
    }

    // --- Acceptance: fires before the action, and only then ----------------

    /// Acceptance (#1126): the burn is surfaced for the action it is about.
    #[test]
    fn a_burn_fires_for_the_action_it_was_recorded_against() {
        let held = [fresh_burn()];
        let fired = burns_that_fire(&held, &the_call(), now());
        assert_eq!(fired.len(), 1, "the same call again meets its own lesson");
        assert_eq!(fired[0].id, "nm-1");
    }

    /// Acceptance (#1126): a burn is not surfaced by something that merely
    /// mentions it. Naming the burned tool and quoting its outcome inside
    /// another tool's arguments fires nothing, because the match is on the
    /// action and its facets and never on text.
    #[test]
    fn an_action_that_only_mentions_the_burn_does_not_fire_it() {
        let talking_about_it = PendingAction::observe(
            "builtin_knowledge_base_write",
            &args(&[(
                "content",
                text("terminal_run: rm -rf build is a mount point and the host lost its cache"),
            )]),
            &here_and_now(),
        );
        assert!(
            burns_that_fire(&[fresh_burn()], &talking_about_it, now()).is_empty(),
            "writing about a burn is not taking the action it is about"
        );
    }

    /// Acceptance (#1126): the near miss on the action's own arguments. Same
    /// tool, same situation, one argument different - and the burn stays quiet.
    /// This is the over-generalization guard: without it, one bad `rm -rf` in
    /// one directory would interrupt every `rm -rf` anywhere.
    #[test]
    fn a_burn_does_not_fire_when_one_action_argument_differs() {
        let elsewhere = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("rm -rf build")),
                ("cwd", text("/srv/other")),
            ]),
            &here_and_now(),
        );
        assert!(
            burns_that_fire(&[fresh_burn()], &elsewhere, now()).is_empty(),
            "the same command in another directory is another act"
        );
    }

    /// Acceptance (#1126): the near miss on the situation. Same tool, same
    /// arguments, another host - and a burn that has been seen only here stays
    /// quiet there.
    #[test]
    fn a_burn_does_not_fire_when_one_situation_facet_differs() {
        let another_host = PendingAction::observe(
            "terminal_run",
            &args(&[("command", text("rm -rf build")), ("cwd", text("/srv/app"))]),
            &Situation::new()
                .with(SituationField::Host, "laptop")
                .with(SituationField::TimeOfDay, "morning")
                .with(SituationField::Weekday, "thursday"),
        );
        assert!(
            burns_that_fire(&[fresh_burn()], &another_host, now()).is_empty(),
            "one occurrence is evidence about one host, not about every host"
        );
    }

    /// A burn is about a tool. The same arguments on another tool are another
    /// act.
    #[test]
    fn a_burn_does_not_fire_for_a_different_tool() {
        let other_tool = PendingAction::observe(
            "builtin_terminal_preview",
            &args(&[("command", text("rm -rf build")), ("cwd", text("/srv/app"))]),
            &here_and_now(),
        );
        assert!(burns_that_fire(&[fresh_burn()], &other_tool, now()).is_empty());
    }

    // --- Acceptance: decay --------------------------------------------------

    /// Acceptance (#1126): a burn nothing has confirmed within its window stops
    /// being surfaced.
    #[test]
    fn a_burn_not_confirmed_within_its_window_stops_firing() {
        let stale = burn_aged(TimeDelta::days(29));
        assert!(
            stale.is_silent(now()),
            "four weeks without a repeat puts a burn under the silence floor"
        );
        assert!(
            burns_that_fire(&[stale], &the_call(), now()).is_empty(),
            "a decayed burn interrupts nothing"
        );
    }

    /// The boundary the window is stated at: two half-lives is exactly the
    /// floor, and a burn at the floor still fires.
    #[test]
    fn a_burn_at_the_silence_floor_still_fires() {
        let at_the_floor = burn_aged(TimeDelta::days(28));
        assert!(
            (at_the_floor.strength(now()) - SILENCE_FLOOR).abs() < 1e-9,
            "two half-lives is a quarter of full strength"
        );
        assert!(!at_the_floor.is_silent(now()));
        assert_eq!(
            burns_that_fire(&[at_the_floor], &the_call(), now()).len(),
            1
        );
    }

    /// A silent burn is still held, and a fresh occurrence confirms it back to
    /// full rather than starting an unrelated second lesson.
    #[test]
    fn a_silent_burn_is_still_held_and_can_be_confirmed_back() {
        let mut stale = burn_aged(TimeDelta::days(40));
        assert!(stale.is_silent(now()));
        assert!(
            !stale.is_forgotten(now()),
            "silence is not forgetting: the row is still readable"
        );
        stale.last_confirmed_at = now();
        assert!(stale.fires(&the_call(), now()));
    }

    /// Past the forget horizon the store may drop the row.
    #[test]
    fn a_burn_past_the_forget_horizon_is_reapable() {
        assert!(burn_aged(TimeDelta::days(57)).is_forgotten(now()));
        assert!(!burn_aged(TimeDelta::days(55)).is_forgotten(now()));
    }

    /// Ordinary skew between the database's clock and the reader's means
    /// nothing: a stamp a few minutes ahead is a burn confirmed just now.
    #[test]
    fn a_stamp_slightly_ahead_of_the_clock_reads_as_a_fresh_burn() {
        let mut ahead = fresh_burn();
        ahead.last_confirmed_at = now() + TimeDelta::minutes(5);
        assert!((ahead.strength(now()) - FULL_STRENGTH).abs() < f64::EPSILON);
        assert!(!ahead.is_silent(now()));
        assert!(!ahead.is_forgotten(now()));
        assert!(ahead.fires(&the_call(), now()));
    }

    /// A stamp far ahead of the clock is a broken clock, not a fresher lesson.
    ///
    /// Read as fresh it would sit at full strength for as long as the skew
    /// lasted - a burn that can be neither silenced nor reaped, interrupting
    /// work the whole time. That is the fires-when-it-should-not direction, so
    /// it is disbelieved: it scores zero, stays quiet, and is reapable.
    #[test]
    fn a_stamp_far_ahead_of_the_clock_is_disbelieved_rather_than_trusted() {
        let mut broken = fresh_burn();
        broken.last_confirmed_at = now() + TimeDelta::days(3);
        assert_eq!(broken.strength(now()), 0.0);
        assert!(broken.is_silent(now()));
        assert!(
            !broken.fires(&the_call(), now()),
            "a lesson nobody can date must not interrupt an act"
        );
        assert!(
            broken.is_forgotten(now()),
            "and it must not sit in the store forever, since nothing can ever \
             raise it"
        );
    }

    // --- Acceptance: broadening needs a second occurrence -------------------

    /// Acceptance (#1126): one occurrence keeps the scope it was written with.
    /// Widening it takes a second occurrence somewhere else, and then the burn
    /// fires on the axis that second occurrence disproved.
    #[test]
    fn broadening_a_burns_scope_needs_a_second_occurrence() {
        let first = fresh_burn();
        let on_a_laptop = PendingAction::observe(
            "terminal_run",
            &args(&[("command", text("rm -rf build")), ("cwd", text("/srv/app"))]),
            &Situation::new()
                .with(SituationField::Host, "laptop")
                .with(SituationField::TimeOfDay, "evening")
                .with(SituationField::Weekday, "sunday"),
        );

        assert!(
            !first.fires(&on_a_laptop, now()),
            "before the second occurrence the burn is about one host only"
        );

        let mut widened = first.clone();
        widened.scope = first.scope.broadened_against(&on_a_laptop.scope);
        widened.occurrences = 2;

        assert!(
            widened.fires(&on_a_laptop, now()),
            "the second occurrence showed the host was not the cause"
        );
        assert!(
            widened.fires(&the_call(), now()),
            "widening never loses the case that produced the burn"
        );
        assert_eq!(
            widened.scope.get(&Facet::Situation(SituationField::Host)),
            None,
            "the disproved facet is dropped, not kept with a second value"
        );
    }

    /// The identity guard: broadening drops circumstance and never drops an
    /// argument, so no sequence of occurrences can turn a burn about one
    /// command into a burn about the tool.
    #[test]
    fn broadening_never_drops_an_argument_facet() {
        let first = fresh_burn();
        let a_different_command = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("git clean -xdf")),
                ("cwd", text("/srv/other")),
            ]),
            &Situation::new().with(SituationField::Host, "laptop"),
        );

        let widened = first.scope.broadened_against(&a_different_command.scope);
        assert_eq!(
            widened.get(&Facet::Argument("command".to_string())),
            Some("rm -rf build"),
            "the command stays required however often something else fails"
        );
        assert_eq!(
            widened.get(&Facet::Argument("cwd".to_string())),
            Some("/srv/app"),
            "the directory stays required too"
        );
    }

    /// Absence is not disagreement. A second occurrence that says nothing about
    /// a facet leaves it required.
    ///
    /// The bug this guards is the module's own failure mode arriving as its own
    /// remedy: one repeat from a headless path, or from a client reporting
    /// neither a hostname nor a timezone, carries an empty situation - and
    /// reading "absent" as "different" would strip a burn's whole circumstance
    /// and set it firing everywhere, on no evidence at all.
    #[test]
    fn a_facet_the_second_occurrence_says_nothing_about_is_kept() {
        let first = fresh_burn();
        let nothing_reported = PendingAction::observe(
            "terminal_run",
            &args(&[("command", text("rm -rf build")), ("cwd", text("/srv/app"))]),
            &Situation::new(),
        );
        let widened = first.scope.broadened_against(&nothing_reported.scope);
        assert_eq!(
            widened, first.scope,
            "a repeat that observed no circumstance widens nothing"
        );
    }

    /// Two occurrences in the same situation widen nothing: there is no
    /// evidence that any facet was innocent.
    #[test]
    fn a_second_occurrence_in_the_same_situation_widens_nothing() {
        let first = fresh_burn();
        let widened = first.scope.broadened_against(&the_call().scope);
        assert_eq!(widened, first.scope);
    }

    // --- Acceptance: extinction is an overlay -------------------------------

    /// Acceptance (#1126): a burn the correction points at stops firing while
    /// staying wholly readable.
    #[test]
    fn an_extinguished_burn_stops_firing_and_keeps_its_content() {
        let mut burn = fresh_burn();
        burn.superseded_by = Some("nm-2".to_string());
        assert!(
            !burn.fires(&the_call(), now()),
            "an extinguished burn interrupts nothing"
        );
        assert!(
            burn.outcome.contains("mount point"),
            "the original lesson is still there to read"
        );
        assert!(
            (burn.strength(now()) - FULL_STRENGTH).abs() < f64::EPSILON,
            "extinction is not decay: the record is unchanged, only overlaid"
        );
    }

    /// A correction is a row in the same store and never fires on its own.
    #[test]
    fn a_correction_never_fires() {
        let mut correction = fresh_burn();
        correction.id = "nm-2".to_string();
        correction.kind = NegativeMemoryKind::Correction;
        correction.outcome = "the same call succeeded; the mount point was removed".to_string();
        assert!(!correction.fires(&the_call(), now()));
        assert!(burns_that_fire(&[correction], &the_call(), now()).is_empty());
    }

    // --- What a burn records, and what it matches on ---------------------

    /// Acceptance (#1126), and the near miss that matters most: a burn left by
    /// a call with no arguments must not fire on a call that has them.
    ///
    /// This is the widest a burn could ever be - one bad call of a tool
    /// standing for every call of it - and it arrives by an ordinary route: the
    /// model forgets a required argument, the tool refuses, and the lesson is
    /// written. The identity is the arguments, so it holds.
    #[test]
    fn a_burn_from_a_call_with_no_arguments_does_not_fire_on_a_call_that_has_them() {
        let forgot_the_arguments =
            PendingAction::observe("terminal_run", &serde_json::json!({}), &here_and_now());
        let mut burn = fresh_burn();
        burn.fingerprint = forgot_the_arguments.fingerprint.clone();
        burn.scope = forgot_the_arguments.scope.clone();

        assert!(
            burn.fires(&forgot_the_arguments, now()),
            "the burn is about the call that was actually made"
        );
        assert!(
            !burn.fires(&the_call(), now()),
            "and about no other call of that tool"
        );
    }

    /// The same guard where the situation contributes nothing: on a headless
    /// deployment, or from a client that reports no host and no timezone, a
    /// zero-argument burn's scope is empty - and an empty scope matches
    /// everything. The act is what stops it, so the act has to be enough.
    #[test]
    fn a_burn_with_an_empty_scope_still_fires_only_on_its_own_act() {
        let nothing_connected =
            PendingAction::observe("terminal_run", &serde_json::json!({}), &Situation::new());
        assert!(
            nothing_connected.scope.is_empty(),
            "the case this test is about: nothing to match on but the act"
        );
        let mut burn = fresh_burn();
        burn.fingerprint = nothing_connected.fingerprint.clone();
        burn.scope = nothing_connected.scope.clone();

        assert!(burn.fires(&nothing_connected, now()));
        assert!(
            !burn.fires(
                &PendingAction::observe(
                    "terminal_run",
                    &args(&[("command", text("rm -rf build"))]),
                    &Situation::new(),
                ),
                now()
            ),
            "an empty scope must not become an empty condition"
        );
    }

    /// Adding an argument makes it another call, so it is another lesson.
    /// Stricter than it needs to be, deliberately: a burn that fails to fire
    /// costs one repeat of a mistake, and a burn that fires on the wrong act
    /// costs something nobody can see.
    #[test]
    fn a_call_that_adds_an_argument_is_a_different_act() {
        let with_a_timeout = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("rm -rf build")),
                ("cwd", text("/srv/app")),
                ("timeout_seconds", serde_json::json!(30)),
            ]),
            &here_and_now(),
        );
        assert!(burns_that_fire(&[fresh_burn()], &with_a_timeout, now()).is_empty());
    }

    /// A structured argument cannot be shown in a warning, and it is still read
    /// whole by the identity. So a call the record cannot describe is still a
    /// call the match can tell from another.
    #[test]
    fn a_structured_argument_is_not_recorded_and_still_identifies_the_call() {
        let one = PendingAction::observe(
            "builtin_file_write",
            &args(&[("blocks", serde_json::json!([{"a": 1}]))]),
            &here_and_now(),
        );
        let other = PendingAction::observe(
            "builtin_file_write",
            &args(&[("blocks", serde_json::json!([{"a": 2}]))]),
            &here_and_now(),
        );
        assert_eq!(
            one.scope.get(&Facet::Argument("blocks".to_string())),
            None,
            "there is nothing short and scalar to record"
        );
        assert_ne!(
            one.fingerprint, other.fingerprint,
            "and the two calls are still two acts"
        );
    }

    /// The bug this guards: two long commands in one directory. The recorded
    /// facets are identical - both commands are too long to hold - so an
    /// identity taken over the record alone would fold two lessons into one and
    /// widen the survivor. The identity reads the argument at full length.
    #[test]
    fn two_arguments_differing_past_the_recorded_length_are_two_acts() {
        let prefix = "x".repeat(MAX_FACET_VALUE_CHARS);
        let one = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text(&format!("{prefix}a"))),
                ("cwd", text("/srv/app")),
            ]),
            &here_and_now(),
        );
        let other = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text(&format!("{prefix}b"))),
                ("cwd", text("/srv/app")),
            ]),
            &here_and_now(),
        );
        assert_eq!(
            one.scope, other.scope,
            "the record cannot tell them apart, which is the case this is about"
        );
        assert_ne!(one.fingerprint, other.fingerprint, "the identity can");
    }

    /// An oversized argument beside a usable one is dropped from the record and
    /// the rest still describes the call. That is why a file write is shown by
    /// its path and not by its contents.
    #[test]
    fn an_oversized_argument_is_dropped_from_the_record_and_the_others_remain() {
        let pending = PendingAction::observe(
            "builtin_file_write",
            &args(&[
                ("path", text("/srv/app/config.toml")),
                ("content", text(&"x".repeat(MAX_FACET_VALUE_CHARS + 1))),
            ]),
            &here_and_now(),
        );
        assert_eq!(
            pending.scope.get(&Facet::Argument("path".to_string())),
            Some("/srv/app/config.toml")
        );
        assert_eq!(
            pending.scope.get(&Facet::Argument("content".to_string())),
            None,
            "a value too long to hold whole is not recorded"
        );
    }

    /// A value with a control character in it cannot be stored whole, so it is
    /// not recorded - and the identity still reads it, so the burn is no wider
    /// for that.
    #[test]
    fn a_control_character_keeps_a_value_out_of_the_record_but_not_the_identity() {
        let one = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("printf a\u{0}b")),
                ("cwd", text("/srv/app")),
            ]),
            &here_and_now(),
        );
        let other = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("printf a\u{0}c")),
                ("cwd", text("/srv/app")),
            ]),
            &here_and_now(),
        );
        assert_eq!(
            one.scope.get(&Facet::Argument("command".to_string())),
            None,
            "nothing a text column cannot hold reaches the record"
        );
        assert_ne!(one.fingerprint, other.fingerprint);
    }

    /// More arguments than the record keeps: the extra ones are dropped in name
    /// order, which is stable, and the identity is unaffected.
    #[test]
    fn a_call_with_more_arguments_than_are_recorded_keeps_the_first_by_name() {
        let many: Vec<(&str, serde_json::Value)> = [
            "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", "a11", "a12",
            "a13",
        ]
        .iter()
        .map(|n| (*n, text("v")))
        .collect();
        assert_eq!(many.len(), MAX_ACTION_FACETS + 1);
        let pending = PendingAction::observe("t", &args(&many), &Situation::new());
        assert_eq!(pending.scope.len(), MAX_ACTION_FACETS);
        assert_eq!(
            pending.scope.get(&Facet::Argument("a13".to_string())),
            None,
            "the last by name is the one dropped"
        );
    }

    /// Arguments that are not an object name nothing, so nothing is recorded -
    /// and they still identify the call.
    #[test]
    fn arguments_that_are_not_an_object_still_identify_the_call() {
        let bare = PendingAction::observe("t", &serde_json::json!("bare"), &Situation::new());
        let listed = PendingAction::observe("t", &serde_json::json!([1, 2]), &Situation::new());
        let nothing = PendingAction::observe("t", &serde_json::Value::Null, &Situation::new());
        assert!(bare.scope.is_empty());
        assert_ne!(bare.fingerprint, listed.fingerprint);
        assert_ne!(bare.fingerprint, nothing.fingerprint);
    }

    // --- Identity -----------------------------------------------------------

    /// The fingerprint is the burn's identity, and identity is arguments alone:
    /// the situation is what widens, so it cannot also be what identifies.
    #[test]
    fn the_fingerprint_reads_the_arguments_and_ignores_the_situation() {
        let here = the_call();
        let elsewhere = PendingAction::observe(
            "terminal_run",
            &args(&[("command", text("rm -rf build")), ("cwd", text("/srv/app"))]),
            &Situation::new().with(SituationField::Host, "laptop"),
        );
        assert_eq!(
            here.fingerprint, elsewhere.fingerprint,
            "the same act in another place is the same act"
        );
    }

    /// Different arguments are different acts, so they carry different
    /// fingerprints and never confirm each other.
    #[test]
    fn different_arguments_produce_different_fingerprints() {
        let here = the_call();
        let other = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("rm -rf build")),
                ("cwd", text("/srv/other")),
            ]),
            &here_and_now(),
        );
        assert_ne!(here.fingerprint, other.fingerprint);
    }

    /// The digest does not depend on the order arguments arrive in.
    #[test]
    fn the_fingerprint_does_not_depend_on_argument_order() {
        let one = PendingAction::observe(
            "t",
            &args(&[("a", text("1")), ("b", text("2"))]),
            &Situation::new(),
        );
        let other = PendingAction::observe(
            "t",
            &args(&[("b", text("2")), ("a", text("1"))]),
            &Situation::new(),
        );
        assert_eq!(one.fingerprint, other.fingerprint);
    }

    /// Two argument sets that differ only in where one name ends and its value
    /// begins must not collide, so each string is written with its length in
    /// front of it - including one carrying whatever byte a separator-only
    /// framing would have leaned on being absent.
    #[test]
    fn argument_names_and_values_cannot_be_confused_for_each_other() {
        let one = PendingAction::observe(
            "t",
            &args(&[("ab", text("c")), ("d", text("e"))]),
            &Situation::new(),
        );
        let other = PendingAction::observe(
            "t",
            &args(&[("a", text("bc")), ("d", text("e"))]),
            &Situation::new(),
        );
        assert_ne!(one.fingerprint, other.fingerprint);

        // The same question where the name itself carries a separator byte.
        let framed =
            PendingAction::observe("t", &args(&[("a\u{1f}b", text("c"))]), &Situation::new());
        let split = PendingAction::observe(
            "t",
            &args(&[("a", text("\u{1f}b\u{1f}c"))]),
            &Situation::new(),
        );
        assert_ne!(framed.fingerprint, split.fingerprint);
    }

    /// The digest must read every argument a parser will hand it. Anything
    /// shallower than the parser's own limit and two calls differing only below
    /// the cut share a fingerprint - one burn firing on an act it was never
    /// recorded against, which is the identity widening.
    ///
    /// Held against the parser in the tree, not against a number written beside
    /// the constant: a parser that grows a deeper limit fails here rather than
    /// quietly folding two acts into one.
    #[test]
    fn the_digest_reads_every_argument_a_parser_will_accept() {
        /// `{"k":{"k":...{"k":<leaf>}}}`, `depth` objects deep.
        fn nested(depth: usize, leaf: &str) -> String {
            let mut text = format!("\"{leaf}\"");
            for _ in 0..depth {
                text = format!("{{\"k\":{text}}}");
            }
            text
        }

        let deepest_accepted = (1..=MAX_ARGUMENT_DEPTH + 8)
            .take_while(|d| serde_json::from_str::<serde_json::Value>(&nested(*d, "a")).is_ok())
            .last()
            .expect("a one-level value parses");
        assert!(
            deepest_accepted <= MAX_ARGUMENT_DEPTH,
            "the parser accepts {deepest_accepted} levels and the digest reads \
             {MAX_ARGUMENT_DEPTH}; two calls differing below that would be one act"
        );

        let one: serde_json::Value = serde_json::from_str(&nested(deepest_accepted, "a"))
            .expect("the deepest accepted value parses");
        let other: serde_json::Value =
            serde_json::from_str(&nested(deepest_accepted, "b")).expect("and so does its twin");
        assert_ne!(
            fingerprint(&one),
            fingerprint(&other),
            "two acts differing only at the deepest level a parser allows must \
             not share one lesson"
        );
    }

    /// What the bound is left doing: a value built in code rather than parsed
    /// has no depth limit of its own, and the digest stops rather than
    /// recursing until the stack runs out. Two such values differing only below
    /// the cut do fold into one act - stated here because it is real, and
    /// unreachable from a tool call.
    #[test]
    fn a_value_deeper_than_a_parser_allows_is_read_to_the_bound_and_no_further() {
        fn built(depth: usize, leaf: &str) -> serde_json::Value {
            let mut value = serde_json::Value::String(leaf.to_string());
            for _ in 0..depth {
                value = serde_json::json!({ "k": value });
            }
            value
        }
        let one = built(MAX_ARGUMENT_DEPTH + 4, "a");
        let other = built(MAX_ARGUMENT_DEPTH + 4, "b");
        assert_eq!(
            fingerprint(&one),
            fingerprint(&other),
            "past the bound the digest stops, and it stops rather than overflowing"
        );
    }

    // --- Stored form --------------------------------------------------------

    /// A facet this build cannot name refuses the whole scope, and the reader
    /// then refuses the memory.
    ///
    /// Dropping the unreadable facet is the tempting answer and it is the
    /// dangerous one: the burn would lose a requirement and fire on acts it had
    /// never been seen with. This is not hypothetical - a database written by a
    /// build that knows one more situation dimension, read by one that does
    /// not, is an ordinary rollback.
    #[test]
    fn a_stored_facet_this_build_cannot_name_refuses_the_whole_scope() {
        assert_eq!(
            Scope::from_stored([
                ("argument", "cwd", "/srv/app"),
                ("situation", "host", "workshop"),
                ("situation", "moon_phase", "waxing"),
            ]),
            None,
            "a scope with an unreadable requirement is no scope at all"
        );
        let readable = Scope::from_stored([
            ("argument", "cwd", "/srv/app"),
            ("situation", "host", "workshop"),
        ])
        .expect("every facet is one this build names");
        assert_eq!(readable.len(), 2);
    }

    /// Every facet kind and every known situation field survives a round trip
    /// through the stored form.
    #[test]
    fn a_scope_round_trips_through_its_stored_form() {
        let original = the_call().scope;
        let rows: Vec<(String, String, String)> = original
            .iter()
            .map(|(f, v)| (f.kind().to_string(), f.name().to_string(), v.to_string()))
            .collect();
        assert_eq!(Scope::from_stored(rows), Some(original));
    }

    /// The stored kind of a memory round trips, and an unknown one is refused.
    #[test]
    fn a_memory_kind_round_trips_and_an_unknown_one_is_refused() {
        for kind in [NegativeMemoryKind::Burn, NegativeMemoryKind::Correction] {
            assert_eq!(NegativeMemoryKind::from_stored(kind.as_str()), Some(kind));
        }
        assert_eq!(NegativeMemoryKind::from_stored("phobia"), None);
    }

    // --- The warning --------------------------------------------------------

    /// The warning says what happened and reads as a candidate to check, in the
    /// same terms a surfaced procedure does, rather than as a refusal.
    #[test]
    fn the_warning_states_the_outcome_and_reads_as_a_candidate() {
        let held = [fresh_burn()];
        let fired = burns_that_fire(&held, &the_call(), now());
        let warning = render_warning(&fired, now()).expect("a fired burn renders a warning");
        assert!(warning.contains("mount point"), "it says what went wrong");
        assert!(
            warning.contains("not a refusal"),
            "it is a candidate, not an instruction"
        );
        assert!(
            warning.contains("make the same call again"),
            "it says how to proceed anyway"
        );
        assert!(
            warning.contains("Called with: command=rm -rf build, cwd=/srv/app"),
            "it says which call it is about, so the model can judge the fit"
        );
        assert!(
            warning.contains("It went badly at: host=workshop"),
            "and it says the circumstance apart from the arguments, so an \
             argument named `host` cannot read like the situation field"
        );
    }

    /// Nothing fired means nothing is rendered: a decision point with no lesson
    /// behind it costs the model no reading at all.
    #[test]
    fn no_fired_burn_renders_no_warning() {
        assert!(render_warning(&[], now()).is_none());
    }

    /// A wall of past failures is not a warning, so the rendered block is
    /// capped. The match is not: every fired lesson comes back, because the
    /// other caller extinguishes what a success disproved, and a lesson left
    /// standing for falling off the end of a warning is a lesson nothing could
    /// ever correct.
    #[test]
    fn every_fired_burn_is_returned_and_the_warning_names_only_a_few() {
        let held: Vec<NegativeMemory> = (0..MAX_WARNED_BURNS + 2)
            .map(|i| {
                let mut burn = burn_aged(TimeDelta::days(i as i64));
                burn.id = format!("nm-{i}");
                burn
            })
            .collect();
        let fired = burns_that_fire(&held, &the_call(), now());
        assert_eq!(
            fired.len(),
            MAX_WARNED_BURNS + 2,
            "every lesson this act disproves has to be reachable"
        );
        assert_eq!(
            fired[0].id, "nm-0",
            "the strongest - most recently confirmed - leads"
        );
        let warning = render_warning(&fired, now()).expect("a fired burn renders a warning");
        assert_eq!(
            warning.matches("\n- Last ").count(),
            MAX_WARNED_BURNS,
            "and only a few of them are put in front of the model"
        );
    }

    /// An outcome longer than the store keeps is cut on a character boundary,
    /// so a multi-byte error message cannot panic the write path.
    #[test]
    fn an_oversized_outcome_is_clamped_on_a_character_boundary() {
        let long = "e\u{301}".repeat(MAX_OUTCOME_CHARS);
        let clamped = clamp_outcome(&long);
        assert_eq!(clamped.chars().count(), MAX_OUTCOME_CHARS);
        assert_eq!(clamp_outcome("  spaced  "), "spaced");
    }
}
