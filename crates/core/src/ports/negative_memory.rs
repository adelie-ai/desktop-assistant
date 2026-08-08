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

use crate::CoreError;
use crate::domain::negative_memory::{NegativeMemory, Scope};

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
    /// The row now holding this lesson.
    pub id: String,
    /// How many times it has been recorded. One means this write created it.
    pub occurrences: u32,
    /// Situation facets this occurrence dropped, because the failure happened
    /// without them. Zero on a first write, because there is nothing to widen.
    pub widened_by: usize,
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

/// Boxed async closure writing a correction over burns that stopped applying.
/// Args: `(burn_ids, note)`.
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
