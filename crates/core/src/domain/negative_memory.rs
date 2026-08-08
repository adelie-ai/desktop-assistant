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
//! An action, and the facets it was taken with. The action is a tool call,
//! because a tool call is the one thing in a turn with an identity that can be
//! matched exactly - and exact matching is what holds the scope narrow. The
//! facets are of two kinds ([`Facet`]), and they are not interchangeable:
//!
//! - **Argument facets** say *what* was done. They are the burn's identity:
//!   [`Scope::fingerprint`] is taken over these alone, and
//!   [`Scope::broadened_against`] refuses to drop one. Two failures of one tool
//!   with different arguments are two lessons, never one.
//! - **Situation facets** say *when and where* it was done - the cue of
//!   [`crate::domain::situation`]. These are what a second occurrence drops. A
//!   burn seen on one host and then on another has shown that the host was not
//!   the cause, so the host stops being required.
//!
//! That is the whole of "scope tightly, broaden only on re-confirmation": the
//! evidence for widening is a repeat somewhere else, and nothing else widens
//! anything.
//!
//! ## Firing is a subset test, confirming is an identity test
//!
//! Two different questions, deliberately answered differently.
//!
//! [`NegativeMemory::fires`] asks whether everything the burn requires is true
//! of the call about to be made. A call that carries *more* than the burn
//! requires still fires it: adding an argument does not make it a different
//! act.
//!
//! Confirmation asks whether this failure is the same lesson as one already
//! held, and that is an exact match on [`Scope::fingerprint`]. A looser rule
//! would fold two lessons into one and widen the survivor, which is the
//! failure mode above arriving by the back door.
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

/// Days after which a burn nothing has confirmed is dropped from the store.
///
/// Four half-lives, so a sixteenth of full strength and twice as long as
/// silence. Deleting it loses nothing a reader would act on and bounds the
/// table without a sweep: the writer reaps on its own path, the way the
/// situation record's writer bounds itself.
pub const FORGET_DAYS: f64 = 56.0;

/// Most facets one action may be scoped by.
///
/// Not a trim - a call carrying more capturable arguments than this is declined
/// outright, by [`PendingAction::observe`]. Keeping only some of them would
/// scope the burn on a subset of what actually happened, which is a wider burn
/// wearing a narrow one's evidence. Twelve is past every tool this workspace
/// advertises.
pub const MAX_ACTION_FACETS: usize = 12;

/// Longest argument value that can be a facet.
///
/// The same bound the situation record uses for its own values, for the same
/// reason: a facet is matched by equality, so it has to be small enough to
/// store whole. A longer value is not truncated to fit - two different values
/// sharing a prefix would then match - it is simply not a facet, which makes
/// the burn wider along an axis nothing could have matched on anyway.
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
/// what bounds that read. Well past what the reap at [`FORGET_DAYS`] leaves
/// behind for any ordinary history.
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

    /// Rebuild a scope from stored `(kind, name, value)` triples, skipping any
    /// facet this build cannot name.
    pub fn from_stored<I, K, N, V>(rows: I) -> Self
    where
        I: IntoIterator<Item = (K, N, V)>,
        K: AsRef<str>,
        N: AsRef<str>,
        V: Into<String>,
    {
        Self(
            rows.into_iter()
                .filter_map(|(kind, name, value)| {
                    Facet::from_stored(kind.as_ref(), name.as_ref()).map(|f| (f, value.into()))
                })
                .collect(),
        )
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
    /// A subset test, not an equality test: a call that carries more than the
    /// burn requires is still the act the burn is about. See the module header
    /// for why confirmation asks a different question.
    pub fn matches(&self, present: &Scope) -> bool {
        self.0
            .iter()
            .all(|(facet, value)| present.get(facet) == Some(value.as_str()))
    }

    /// This scope widened by a second occurrence.
    ///
    /// A situation facet whose observed value differs is dropped: the failure
    /// happened without it, so it was not the cause. Identity facets are kept
    /// whatever `observed` says, because a differing argument means a different
    /// lesson and this is the wrong operation for it - see the module header.
    ///
    /// This is the only thing in the module that makes a burn wider.
    #[must_use]
    pub fn broadened_against(&self, observed: &Scope) -> Scope {
        Scope(
            self.0
                .iter()
                .filter(|(facet, value)| {
                    facet.is_identity() || observed.get(facet) == Some(value.as_str())
                })
                .map(|(f, v)| (f.clone(), v.clone()))
                .collect(),
        )
    }

    /// A stable digest of the identity facets alone: the burn's handle.
    ///
    /// Two failures share a lesson when their actions and their fingerprints
    /// both match. The situation is excluded on purpose - it is what widens,
    /// so it cannot also be what identifies.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        for (facet, value) in self.0.iter().filter(|(f, _)| f.is_identity()) {
            hasher.update(facet.name().as_bytes());
            hasher.update([0x1f]);
            hasher.update(value.as_bytes());
            hasher.update([0x1e]);
        }
        format!("{:x}", hasher.finalize())[..32].to_string()
    }
}

/// A tool call about to be made, reduced to what a burn can be matched on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingAction {
    /// The tool's name.
    pub action: String,
    /// Everything observed about the call and the moment it is made in.
    pub scope: Scope,
}

impl PendingAction {
    /// Read a pending call and the present situation into a matchable action.
    ///
    /// `None` when the call cannot be scoped, and a declined call neither fires
    /// a burn nor writes one. Three cases, all of them the same refusal to
    /// over-generalize:
    ///
    /// - `arguments` is neither an object nor null. There is nothing to name a
    ///   facet by.
    /// - The call carries arguments but none of them can be a facet - every one
    ///   is structured, or longer than [`MAX_FACET_VALUE_CHARS`]. A burn
    ///   written here would be keyed on the bare tool name and would interrupt
    ///   every later call to it.
    /// - The call carries more than [`MAX_ACTION_FACETS`] usable arguments.
    ///   Keeping some of them would scope the burn on less than what happened.
    ///
    /// An argument that is structured or too long is dropped while others
    /// remain; that widens the burn along an axis equality could not have
    /// matched anyway, and it is why a file write is remembered by its path
    /// rather than by its contents.
    pub fn observe(
        action: impl Into<String>,
        arguments: &serde_json::Value,
        situation: &Situation,
    ) -> Option<Self> {
        let arguments = match arguments {
            serde_json::Value::Null => None,
            serde_json::Value::Object(map) => Some(map),
            _ => return None,
        };

        let mut scope = Scope::new();
        if let Some(map) = arguments {
            let usable: Vec<(&String, String)> = map
                .iter()
                .filter_map(|(name, value)| facet_value(value).map(|v| (name, v)))
                .collect();
            if !map.is_empty() && usable.is_empty() {
                return None;
            }
            if usable.len() > MAX_ACTION_FACETS {
                return None;
            }
            for (name, value) in usable {
                scope = scope.with(Facet::Argument(name.clone()), value);
            }
        }
        for (field, value) in situation.iter() {
            scope = scope.with(Facet::Situation(field), value);
        }

        Some(Self {
            action: action.into(),
            scope,
        })
    }

    /// The identity of this action: what a confirmation must match exactly.
    pub fn fingerprint(&self) -> String {
        self.scope.fingerprint()
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
        let elapsed = now.signed_duration_since(self.last_confirmed_at);
        let days = elapsed.num_milliseconds() as f64 / (1000.0 * 60.0 * 60.0 * 24.0);
        if days <= 0.0 {
            return FULL_STRENGTH;
        }
        FULL_STRENGTH * 0.5_f64.powf(days / HALF_LIFE_DAYS)
    }

    /// Whether this has decayed past the point of interrupting anything.
    pub fn is_silent(&self, now: DateTime<Utc>) -> bool {
        self.strength(now) < SILENCE_FLOOR
    }

    /// Whether this should be dropped from the store.
    pub fn is_forgotten(&self, now: DateTime<Utc>) -> bool {
        now.signed_duration_since(self.last_confirmed_at)
            .num_milliseconds() as f64
            / (1000.0 * 60.0 * 60.0 * 24.0)
            > FORGET_DAYS
    }

    /// Whether this burn interrupts `pending`.
    ///
    /// Five conditions, and every one of them is a way the burn could be wrong
    /// about right now: it must be a lesson rather than a correction, it must
    /// not have been extinguished, it must be about this tool, it must still be
    /// loud enough, and everything it requires must hold.
    pub fn fires(&self, pending: &PendingAction, now: DateTime<Utc>) -> bool {
        self.kind == NegativeMemoryKind::Burn
            && self.superseded_by.is_none()
            && self.action == pending.action
            && !self.is_silent(now)
            && self.scope.matches(&pending.scope)
    }
}

/// The burns that interrupt `pending`, strongest first.
///
/// Bounded by [`MAX_WARNED_BURNS`]: the caller is about to put this in front of
/// a model that has to read all of it before it can act.
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
    fired.truncate(MAX_WARNED_BURNS);
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
    for burn in fired {
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
        if !burn.scope.is_empty() {
            out.push_str(&format!(
                "  It went badly with: {}\n",
                describe(&burn.scope)
            ));
        }
    }
    out.push_str(
        "\nDecide whether the cause still applies. If it does not - the fault is fixed, the \
         interface changed, or you mean something different this time - make the same call \
         again and it will run. If it does, take another way.",
    );
    Some(out)
}

/// One line naming what a scope requires, for the warning above.
fn describe(scope: &Scope) -> String {
    scope
        .iter()
        .map(|(facet, value)| format!("{}={value}", facet.name()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The stored form of an argument value, when it can be one.
///
/// Scalars only, and only short ones. See [`MAX_FACET_VALUE_CHARS`] for why a
/// long value is dropped rather than cut down.
fn facet_value(value: &serde_json::Value) -> Option<String> {
    let rendered = match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => return None,
    };
    (rendered.chars().count() <= MAX_FACET_VALUE_CHARS).then_some(rendered)
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
        .expect("a two-argument call is scopeable")
    }

    /// The burn that call left behind, recorded `age` before the fixed clock.
    fn burn_aged(age: TimeDelta) -> NegativeMemory {
        let pending = the_call();
        NegativeMemory {
            id: "nm-1".to_string(),
            action: pending.action.clone(),
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
        )
        .expect("a one-argument call is scopeable");
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
        )
        .expect("a two-argument call is scopeable");
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
        )
        .expect("a two-argument call is scopeable");
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
        )
        .expect("a two-argument call is scopeable");
        assert!(burns_that_fire(&[fresh_burn()], &other_tool, now()).is_empty());
    }

    /// Firing is a subset test: a call that carries more than the burn requires
    /// is still the act the burn is about.
    #[test]
    fn a_call_that_carries_more_than_the_burn_requires_still_fires_it() {
        let with_an_extra_argument = PendingAction::observe(
            "terminal_run",
            &args(&[
                ("command", text("rm -rf build")),
                ("cwd", text("/srv/app")),
                ("timeout_seconds", serde_json::json!(30)),
            ]),
            &here_and_now(),
        )
        .expect("a three-argument call is scopeable");
        assert_eq!(
            burns_that_fire(&[fresh_burn()], &with_an_extra_argument, now()).len(),
            1,
            "adding an argument does not make it a different act"
        );
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

    /// A clock that runs backwards must not read as a strengthened burn or a
    /// silent one.
    #[test]
    fn a_burn_confirmed_in_the_future_holds_full_strength() {
        let mut ahead = fresh_burn();
        ahead.last_confirmed_at = now() + TimeDelta::days(3);
        assert!((ahead.strength(now()) - FULL_STRENGTH).abs() < f64::EPSILON);
        assert!(!ahead.is_forgotten(now()));
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
        )
        .expect("a two-argument call is scopeable");

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
        )
        .expect("a two-argument call is scopeable");

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

    // --- Scoping: what can and cannot become a burn -------------------------

    /// A call whose every argument is structured or oversized cannot be scoped,
    /// so nothing is learned from it. A burn keyed on the bare tool name would
    /// interrupt every later call to that tool, which is the failure mode this
    /// whole design exists to prevent.
    #[test]
    fn a_call_whose_arguments_cannot_be_scoped_is_declined() {
        assert!(
            PendingAction::observe(
                "builtin_file_write",
                &args(&[("blocks", serde_json::json!([{"a": 1}]))]),
                &here_and_now(),
            )
            .is_none(),
            "a structured-only call cannot be scoped, so it is not learned from"
        );

        let too_long = "x".repeat(MAX_FACET_VALUE_CHARS + 1);
        assert!(
            PendingAction::observe(
                "builtin_file_write",
                &args(&[("content", text(&too_long))]),
                &here_and_now(),
            )
            .is_none(),
            "an oversized-only call cannot be scoped either"
        );
    }

    /// A call with more usable arguments than can be scoped is declined rather
    /// than scoped on a subset - a subset is a wider burn wearing a narrower
    /// one's evidence.
    #[test]
    fn a_call_with_more_arguments_than_can_be_scoped_is_declined() {
        let many: Vec<(&str, serde_json::Value)> = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13",
        ]
        .iter()
        .map(|n| (*n, text("v")))
        .collect();
        assert_eq!(many.len(), MAX_ACTION_FACETS + 1);
        assert!(PendingAction::observe("t", &args(&many), &here_and_now()).is_none());
    }

    /// An oversized argument beside a usable one is dropped, and the burn is
    /// keyed on what is left. That is why a file write is remembered by its
    /// path and not by its contents.
    #[test]
    fn an_oversized_argument_is_dropped_while_the_others_still_scope_the_burn() {
        let pending = PendingAction::observe(
            "builtin_file_write",
            &args(&[
                ("path", text("/srv/app/config.toml")),
                ("content", text(&"x".repeat(MAX_FACET_VALUE_CHARS + 1))),
            ]),
            &here_and_now(),
        )
        .expect("one usable argument is enough to scope the call");
        assert_eq!(
            pending.scope.get(&Facet::Argument("path".to_string())),
            Some("/srv/app/config.toml")
        );
        assert_eq!(
            pending.scope.get(&Facet::Argument("content".to_string())),
            None,
            "a value too long to match on is not a facet"
        );
    }

    /// A tool that takes no arguments is scopeable: its name is the whole
    /// identity, and that is as narrow as the act gets.
    #[test]
    fn a_call_with_no_arguments_is_scoped_by_its_name_and_situation() {
        let pending = PendingAction::observe(
            "builtin_tasks_list",
            &serde_json::json!({}),
            &here_and_now(),
        )
        .expect("a no-argument call is scopeable");
        assert_eq!(pending.scope.len(), 3, "the three situation facets");
        assert!(
            PendingAction::observe(
                "builtin_tasks_list",
                &serde_json::Value::Null,
                &here_and_now()
            )
            .is_some(),
            "null arguments read the same as an empty object"
        );
    }

    /// Arguments that are not an object at all name nothing, so they cannot be
    /// scoped.
    #[test]
    fn a_call_whose_arguments_are_not_an_object_is_declined() {
        assert!(PendingAction::observe("t", &serde_json::json!("bare"), &here_and_now()).is_none());
        assert!(PendingAction::observe("t", &serde_json::json!([1, 2]), &here_and_now()).is_none());
    }

    /// A deployment with nothing connected reports an empty situation. The burn
    /// is then scoped by its arguments alone, which is narrower than nothing
    /// and is the most the system can honestly say.
    #[test]
    fn a_call_in_an_empty_situation_is_scoped_by_its_arguments_alone() {
        let pending = PendingAction::observe(
            "terminal_run",
            &args(&[("command", text("rm -rf build"))]),
            &Situation::new(),
        )
        .expect("an empty situation does not stop a call being scoped");
        assert_eq!(pending.scope.len(), 1);
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
        )
        .expect("a two-argument call is scopeable");
        assert_eq!(
            here.fingerprint(),
            elsewhere.fingerprint(),
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
        )
        .expect("a two-argument call is scopeable");
        assert_ne!(here.fingerprint(), other.fingerprint());
    }

    /// The digest does not depend on the order arguments arrive in.
    #[test]
    fn the_fingerprint_does_not_depend_on_argument_order() {
        let one = PendingAction::observe(
            "t",
            &args(&[("a", text("1")), ("b", text("2"))]),
            &Situation::new(),
        )
        .expect("scopeable");
        let other = PendingAction::observe(
            "t",
            &args(&[("b", text("2")), ("a", text("1"))]),
            &Situation::new(),
        )
        .expect("scopeable");
        assert_eq!(one.fingerprint(), other.fingerprint());
    }

    /// Two argument sets that differ only in where a separator falls must not
    /// collide, so the digest frames each name and value.
    #[test]
    fn argument_names_and_values_cannot_be_confused_for_each_other() {
        let one = PendingAction::observe(
            "t",
            &args(&[("ab", text("c")), ("d", text("e"))]),
            &Situation::new(),
        )
        .expect("scopeable");
        let other = PendingAction::observe(
            "t",
            &args(&[("a", text("bc")), ("d", text("e"))]),
            &Situation::new(),
        )
        .expect("scopeable");
        assert_ne!(one.fingerprint(), other.fingerprint());
    }

    // --- Stored form --------------------------------------------------------

    /// A facet this build cannot name is dropped on read rather than kept as an
    /// unmatchable requirement, because a requirement nothing can satisfy would
    /// silence the burn instead of narrowing it.
    #[test]
    fn a_stored_facet_this_build_cannot_name_is_dropped() {
        let scope = Scope::from_stored([
            ("argument", "cwd", "/srv/app"),
            ("situation", "host", "workshop"),
            ("situation", "moon_phase", "waxing"),
            ("astrology", "sign", "leo"),
        ]);
        assert_eq!(scope.len(), 2);
        assert_eq!(
            scope.get(&Facet::Situation(SituationField::Host)),
            Some("workshop")
        );
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
        assert_eq!(Scope::from_stored(rows), original);
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
            warning.contains("command=rm -rf build"),
            "it says what it is scoped to, so the model can judge the fit"
        );
    }

    /// Nothing fired means nothing is rendered: a decision point with no lesson
    /// behind it costs the model no reading at all.
    #[test]
    fn no_fired_burn_renders_no_warning() {
        assert!(render_warning(&[], now()).is_none());
    }

    /// A wall of past failures is not a warning. The set is capped, strongest
    /// first, so the model reads the most recent lessons and not all of them.
    #[test]
    fn the_warning_names_at_most_the_capped_number_of_burns() {
        let held: Vec<NegativeMemory> = (0..MAX_WARNED_BURNS + 2)
            .map(|i| {
                let mut burn = burn_aged(TimeDelta::days(i as i64));
                burn.id = format!("nm-{i}");
                burn
            })
            .collect();
        let fired = burns_that_fire(&held, &the_call(), now());
        assert_eq!(fired.len(), MAX_WARNED_BURNS);
        assert_eq!(
            fired[0].id, "nm-0",
            "the strongest - most recently confirmed - leads"
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
