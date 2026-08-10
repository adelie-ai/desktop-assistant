//! Outbound port for negative memory (#1126).
//!
//! [`crate::domain::negative_memory`] states the whole rule a burn follows.
//! This is the three things that rule needs a store for: read what is held,
//! record a bad outcome, and write a correction over a lesson that stopped
//! applying.
//!
//! ## Why the whole live set is read at once
//!
//! A burn is matched at a decision point, and a decision point is every tool
//! call in a turn. A read per call would put a database round trip in front of
//! each one, so the turn reads its user's live burns once and matches them in
//! memory. [`MAX_LIVE_BURNS`] is what bounds that read, and the store's own
//! reap at [`FORGET_DAYS`] is what keeps an ordinary history well under it.
//!
//! The matching itself is not here and is not in the adapter. It is
//! [`burns_that_fire`], one pure function, so there is one place that decides
//! what a burn applies to.
//!
//! [`MAX_LIVE_BURNS`]: crate::domain::negative_memory::MAX_LIVE_BURNS
//! [`FORGET_DAYS`]: crate::domain::negative_memory::FORGET_DAYS
//! [`burns_that_fire`]: crate::domain::negative_memory::burns_that_fire
//!
//! ## Why this is not the knowledge use log
//!
//! The use log ([`crate::ports::knowledge_use`]) answers what happened to an
//! *entry*: it was offered, it was opened, a mark said it was wrong. Every one
//! of its tables carries a foreign key to `knowledge_base(id)`, which is what
//! frees an entry's rows when the entry is reaped.
//!
//! A burn is not about an entry. It is about an act, and an act has no row in
//! that table - the same constraint that gave skills their own log rather than
//! a column on the knowledge one. A negative mark and a burn also answer
//! different questions and are retrieved at different moments: a mark is prune
//! evidence and is never surfaced, and a burn is surfaced before an action and
//! never prunes anything. Setting one does not write the other.
//!
//! ## Recording is not allowed to fail the work it measures
//!
//! Both writes go through
//! [`record_in_background`](crate::ports::knowledge_use::record_in_background),
//! for the reason that function states. A burn that could not be written costs
//! a lesson; a burn that could break a turn costs the turn.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::CoreError;
use crate::domain::negative_memory::{Facet, NegativeMemory, Scope};

/// One bad outcome, as the store is told about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnObservation {
    /// The tool that went badly.
    pub action: String,
    /// The digest of the arguments it went badly with. With `action`, the
    /// identity a later occurrence must match exactly to confirm this lesson
    /// rather than start a second one.
    pub fingerprint: String,
    /// What the burn records and shows: the arguments short enough to hold
    /// whole, and the situation the call was made in.
    pub scope: Scope,
    /// What went wrong, already clamped by
    /// [`clamp_outcome`](crate::domain::negative_memory::clamp_outcome).
    pub outcome: String,
}

/// What recording a bad outcome did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnWrite {
    /// The row now holding this lesson, or empty when nothing was written.
    ///
    /// Empty is a real answer, not an error: the identity this observation
    /// would have taken was extinguished by another writer between the conflict
    /// and the re-read, so there is no live lesson to confirm and the next
    /// occurrence starts one.
    pub id: String,
    /// How many times it has been recorded. One means this write created it,
    /// and zero means nothing was written - see `id`.
    pub occurrences: u32,
    /// Situation facets this occurrence dropped, because the failure happened
    /// without them. Zero on a first write, because there is nothing to widen.
    pub widened_by: usize,
}

/// A circumstance a burn once required and no longer does.
///
/// The only thing in the whole feature that makes a burn wider is a second
/// occurrence dropping the situation facets it disagreed with. So a dropped
/// facet is the one visible trace of over-generalization, and that is why it is
/// kept rather than deleted: a person looking at a burn that fires everywhere
/// needs to see that it started at one host on one morning, and when it stopped
/// asking for that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedFacet {
    /// Which circumstance it was. Always a [`Facet::Situation`] - an argument
    /// facet is the burn's identity and is never dropped.
    pub facet: Facet,
    /// The value the burn was born requiring, which is what says how narrow it
    /// started.
    pub value: String,
    /// When the occurrence that disagreed with it was recorded.
    pub dropped_at: DateTime<Utc>,
}

/// One negative memory read on its own, in full.
///
/// Three things a list row cannot carry and a person deciding whether to clear
/// one needs: what it still requires, what it stopped requiring, and whether
/// anything has already corrected it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnRecord {
    /// The memory itself, live or already corrected. Not filtered on
    /// `superseded_by`: a corrected memory stays readable, which is the whole
    /// reason clearing is an overlay and not a delete.
    pub memory: NegativeMemory,
    /// The circumstances a later occurrence dropped, oldest drop first.
    pub dropped: Vec<DroppedFacet>,
    /// The correction written over it, when one has been.
    pub correction: Option<NegativeMemory>,
}

/// The store behind negative memory.
///
/// Every method is scoped to the current user through
/// [`crate::ports::auth::current_user_id`].
pub trait NegativeMemoryStore: Send + Sync {
    /// Every live burn this user holds, most recently confirmed first.
    ///
    /// Live means a lesson rather than a correction, and not extinguished.
    /// Decay is deliberately *not* applied here: one rule decides what a burn
    /// applies to, and it is in the domain.
    fn live_burns(&self) -> impl Future<Output = Result<Vec<NegativeMemory>, CoreError>> + Send;

    /// Record a bad outcome at full strength.
    ///
    /// Writes a new lesson, or confirms the one already holding this identity -
    /// same action, same argument facets - and widens it by dropping every
    /// situation facet this occurrence disagrees with. A first write widens
    /// nothing, because there is nothing to widen: that is what makes
    /// broadening need a second occurrence.
    ///
    /// Repeating one call with one observation is safe. It moves the
    /// confirmation stamp and the count, and it cannot widen a scope that a
    /// second call with the same facets already agrees with.
    fn record_burn(
        &self,
        observation: BurnObservation,
    ) -> impl Future<Output = Result<BurnWrite, CoreError>> + Send;

    /// Write a correction over each of `ids`, and report which were
    /// extinguished.
    ///
    /// The correction is its own row carrying the burn's action and scope, and
    /// the burn's `superseded_by` names it. Nothing is deleted: "this went
    /// badly, and later it stopped going badly" is knowledge, and deleting it
    /// lets the same lesson be learned again from nothing.
    ///
    /// An id already extinguished, or naming a row this user does not hold,
    /// changes nothing and is not an error.
    fn extinguish(
        &self,
        ids: Vec<String>,
        note: String,
    ) -> impl Future<Output = Result<Vec<String>, CoreError>> + Send;

    /// One lesson read in full, by id, whether or not it has been corrected.
    ///
    /// `None` for an id this user does not hold, which covers both a memory
    /// that was reaped and one another tenant holds - a caller cannot tell the
    /// two apart, and must not be able to. `None` too for a correction's own
    /// id: a correction is the record of a lesson that stopped applying rather
    /// than a lesson, so answering with one would describe a record as though
    /// it were an act being held. It is readable on the burn it corrects.
    ///
    /// The three parts of the answer come from one snapshot, so an occurrence
    /// count and the facets that occurrence dropped can never disagree.
    fn burn(
        &self,
        id: String,
    ) -> impl Future<Output = Result<Option<BurnRecord>, CoreError>> + Send;

    /// Everything ever recorded against `action`: live burns, extinguished
    /// ones, and the corrections written over them.
    ///
    /// The read that proves an overlay did not delete anything.
    fn history(
        &self,
        action: String,
    ) -> impl Future<Output = Result<Vec<NegativeMemory>, CoreError>> + Send;
}

/// Boxed async closure reading this user's live burns, for wiring the store
/// through non-generic boundaries.
pub type LiveBurnsFn = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<Vec<NegativeMemory>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure recording a bad outcome.
pub type RecordBurnFn = Arc<
    dyn Fn(BurnObservation) -> Pin<Box<dyn Future<Output = Result<BurnWrite, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure reading one memory in full, by id.
///
/// The read behind a person's "why will it not do this?", so it answers for a
/// corrected memory as well as a live one.
pub type InspectBurnFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<Option<BurnRecord>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure writing a correction over burns that stopped applying.
/// Args: `(burn_ids, note)`.
///
/// One closure serves both writers, because both write the same thing. A tool
/// call succeeding where it once failed and a person deciding a lesson is wrong
/// are the same event to the store: this lesson stopped applying, and here is
/// the note saying why. A separate person-only path would be a second way to
/// write one row, and the two would drift.
pub type ExtinguishBurnsFn = Arc<
    dyn Fn(
            Vec<String>,
            String,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_store_exists<T: NegativeMemoryStore>() {}
}
