//! Outbound port for the knowledge use log (#698).
//!
//! Three acts are recorded through this port, and one read comes back out of
//! it. What each act means, and why none of them is an inference, is in
//! [`crate::domain::knowledge_use`].
//!
//! ## An open is a taken-up offer
//!
//! [`KnowledgeUseLog::record_opened`] does not count every fetch by id. It
//! counts a fetch of an entry that is standing offered in the same
//! conversation, and it takes that offer down as it counts. Two things follow,
//! and both are the point:
//!
//! - A read that nothing offered - an id from a pinned note, an id the model
//!   held from an earlier task - records nothing. Otherwise ordinary
//!   bookkeeping would inflate the signal that ranking reads.
//! - A second fetch of the same entry in the same turn records one open. The
//!   offer is already down, so the write is idempotent and a retried tool call
//!   is safe.
//!
//! ## An offer stands for one turn
//!
//! A `[Recall]` block is rendered once per turn, from the user's prompt, so an
//! offer made by it **replaces** whatever that conversation had standing. A
//! search happens inside a turn that is already running, so an offer made by
//! one is **added** to what stands. [`OfferSource`] carries which. The effect
//! is that "offered in the same turn" needs no turn identifier: the standing
//! set is this turn's set, because the turn's first block replaced it.
//!
//! The turn records its offer whether or not the block showed anything, and
//! whether or not the lookup succeeded. That is load-bearing rather than tidy:
//! a lookup that timed out, or whose knowledge arm failed, would otherwise
//! leave the previous turn's offers standing, and the model - which still has
//! the previous turn's block in its transcript - could take one up on a later
//! turn. The window would then be "since the last successful lookup" rather
//! than one turn, and it would inflate the highest-quality signal in the log.
//!
//! The one degraded case is a deployment with recall switched off, where the
//! block never renders and nothing replaces the set at a turn boundary. An
//! offer made by a search then stands until it is taken up or until
//! [`MAX_STANDING_OFFERS`] pushes it out. That is a wider window than a turn,
//! never a narrower one, and it still refuses the read that nothing offered.
//!
//! ## Recording never fails a read
//!
//! A use record is a measurement of a read, and a measurement must not be able
//! to break what it measures. Every call site therefore goes through
//! [`record_in_background`], which runs the write off the caller's path and
//! drops its error into a log line.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::knowledge_use::{KnowledgeUseRecord, MarkPolarity, MarkSource};
use crate::domain::situation::{Situation, SituationCue, SituationRecord, SituationSources};
use crate::ports::auth::{current_user_id, with_user_id};
use crate::ports::transport::current_client_context;

/// How many offers one conversation may have standing.
///
/// A `[Recall]` block clears the conversation before it makes its own offers,
/// so on an ordinary deployment a conversation holds one turn's worth and this
/// is never reached. It bounds the case nothing else does: a deployment that
/// renders no block, where a search adds offers and nothing clears them, and a
/// long conversation would otherwise accumulate one row per entry it ever saw.
///
/// The figure is a storage bound, not a ranking coefficient. An offer that has
/// fallen this far behind is one the model is not going to take up.
pub const MAX_STANDING_OFFERS: usize = 256;

/// Which kind of surface put the entries in front of the model.
///
/// The distinction is not descriptive. It decides what happens to the offers
/// that were already standing - see the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfferSource {
    /// The `[Recall]` block, rendered once at the start of a turn. Replaces
    /// the conversation's standing offers.
    Recall,
    /// A knowledge-base search the model ran inside a turn. Adds to the
    /// conversation's standing offers.
    Search,
}

/// Where an offer was made, and by what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferScope {
    /// The conversation the entries were shown in. An open counts only when it
    /// happens in the same conversation.
    pub conversation_id: String,
    /// What showed them.
    pub source: OfferSource,
}

impl OfferScope {
    /// An offer made by the `[Recall]` block of `conversation_id`.
    pub fn recall(conversation_id: impl Into<String>) -> Self {
        Self {
            conversation_id: conversation_id.into(),
            source: OfferSource::Recall,
        }
    }

    /// An offer made by a search inside `conversation_id`.
    pub fn search(conversation_id: impl Into<String>) -> Self {
        Self {
            conversation_id: conversation_id.into(),
            source: OfferSource::Search,
        }
    }
}

/// One request to set a mark on one or more entries.
///
/// A source holds one standing mark per entry, so a second request from the
/// same source replaces the first rather than adding to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkRequest {
    /// The entries to mark. Ids the caller does not own, and ids of retired
    /// entries, are simply not marked.
    pub entry_ids: Vec<String>,
    /// Whether the entries helped or were wrong.
    pub polarity: MarkPolarity,
    /// Who is marking.
    pub source: MarkSource,
    /// Why, in the marker's own words. A negative mark's reason is what makes
    /// the record usable later.
    pub reason: Option<String>,
}

/// What the situation says about one recall lookup (#1125).
///
/// Two answers to one question, read together because they are only meaningful
/// together: a record nothing can grade scores zero, and a cue nothing carries a
/// record for scores zero as well.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SituationSignal {
    /// The situations each candidate has been seen in, by entry id. An id the
    /// table holds nothing for is absent.
    pub records: Vec<(String, SituationRecord)>,
    /// The present situation read against the whole store, or `None` where the
    /// store cannot grade it.
    pub cue: Option<SituationCue>,
}

/// The knowledge use log: what was offered, what was opened, what was marked.
///
/// Every method is scoped to the current user through
/// [`crate::ports::auth::current_user_id`], and every write refuses an entry
/// the caller does not own. Counts returned are rows actually written, so a
/// caller can tell "recorded nothing" from "recorded everything asked".
pub trait KnowledgeUseLog: Send + Sync {
    /// Record that `entry_ids` were put in front of the model, and leave them
    /// standing as offers in `scope.conversation_id`.
    ///
    /// [`OfferSource::Recall`] clears that conversation's standing offers
    /// first, because the block that produced it is rendered once per turn.
    fn record_offered(
        &self,
        scope: OfferScope,
        entry_ids: Vec<String>,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// Record an open for each of `entry_ids` that is standing offered in
    /// `conversation_id`, and take those offers down.
    ///
    /// An id with no standing offer records nothing and is not an error: the
    /// model reads entries for many reasons, and only a taken-up offer is
    /// evidence.
    ///
    /// `situation` is where the open happened, and it is recorded against
    /// exactly the ids that became opens - #238's accumulation rule, carried by
    /// the write that already decides which ids count. Passing
    /// [`Situation::new`] records no situation, which is what a caller with
    /// nothing connected passes.
    fn record_opened(
        &self,
        conversation_id: String,
        entry_ids: Vec<String>,
        situation: Situation,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// Record that `entry_ids` were seen in `situation` (#1125).
    ///
    /// The write path's half of the same rule, for the moment an entry is
    /// observed rather than the moment it is reused. Idempotent by key: a value
    /// the entry's record already holds moves its counters and changes nothing
    /// the ranking reads, which is what stops the retrieve-record-retrieve loop
    /// after one step.
    ///
    /// Entries the caller does not own, retired entries, and an empty situation
    /// all write nothing, and none of them is an error.
    fn record_situation(
        &self,
        entry_ids: Vec<String>,
        situation: Situation,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// Everything one lookup needs of the situation: what each candidate has
    /// been seen in, and what the present situation is worth over this store.
    ///
    /// **One call rather than two, because it is one connection rather than
    /// two.** This read sits on the pre-prompt recall path, which already runs
    /// the pad arm and the use-log read at the same time, and the default
    /// connection pool holds five. Two more concurrent reads per turn would let
    /// one turn hold the whole pool, and a second turn would then wait on
    /// acquisition - which no statement timeout bounds. The two halves also
    /// describe one instant this way, so a fan can never be counted against a
    /// population that has already moved.
    ///
    /// Ids with no record are absent from [`SituationSignal::records`] rather
    /// than returned empty, on the same terms as [`Self::records`]. The cue is
    /// `None` where the store cannot grade one - see [`SituationCue::measured`] -
    /// and the caller then ranks the way it ranked before the cue existed.
    ///
    /// The cue is measured over the source and never over one lookup's
    /// candidates, for the reason [`crate::ports::recall::RecallDispersion`]
    /// states.
    fn situation_signal(
        &self,
        entry_ids: Vec<String>,
        situation: Situation,
    ) -> impl Future<Output = Result<SituationSignal, CoreError>> + Send;

    /// Set the standing mark for `request.source` on each owned entry named,
    /// and report the ids that were marked.
    ///
    /// The ids come back rather than a count, because the caller asked for
    /// this write and has to be able to say which of the ids it named did not
    /// land. An id the caller does not own, and one that names a retired
    /// entry, are both simply absent - the same answer every other read of the
    /// knowledge base gives for the same id.
    fn record_mark(
        &self,
        request: MarkRequest,
    ) -> impl Future<Output = Result<Vec<String>, CoreError>> + Send;

    /// What the log knows about each of `entry_ids`.
    ///
    /// The read that ranking will use. Ids with no record are absent from the
    /// answer rather than returned as zeroes, so a caller can tell an entry
    /// nothing has seen from one that was offered and ignored -
    /// [`KnowledgeUseRecord::unseen`] fills in the first case where a caller
    /// wants one record per id.
    fn records(
        &self,
        entry_ids: Vec<String>,
    ) -> impl Future<Output = Result<Vec<KnowledgeUseRecord>, CoreError>> + Send;
}

/// Boxed async closure that records an offer, for wiring the log through
/// non-generic boundaries.
pub type KnowledgeOfferedFn = Arc<
    dyn Fn(
            OfferScope,
            Vec<String>,
        ) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure that records opens for a conversation's standing
/// offers. Args: `(conversation_id, entry_ids, situation)`.
pub type KnowledgeOpenedFn = Arc<
    dyn Fn(
            String,
            Vec<String>,
            Situation,
        ) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure that records the situation an entry was observed in
/// (#1125). Args: `(entry_ids, situation)`.
pub type KnowledgeSituationFn = Arc<
    dyn Fn(Vec<String>, Situation) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// The situation this request is happening in, read off the clock and the
/// client's own report (#549).
///
/// The one place the wire's field names are mapped onto the domain's, so a
/// write path and a read path cannot disagree about what "the situation" means.
/// [`Situation::observe`] holds every rule; this holds only the mapping and the
/// clock.
///
/// An empty answer is ordinary: a client that reported no context, or a caller
/// outside any request scope, produces one, and every path downstream treats it
/// as "nothing connected".
pub fn current_situation() -> Situation {
    let client = current_client_context();
    let sources = SituationSources {
        host: client.as_ref().and_then(|c| c.hostname.as_deref()),
        timezone: client.as_ref().and_then(|c| c.timezone.as_deref()),
    };
    Situation::observe(chrono::Utc::now(), &sources)
}

/// Boxed async closure that sets a standing mark and reports the ids marked.
pub type KnowledgeMarkFn = Arc<
    dyn Fn(MarkRequest) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Run a use-log write off the caller's path.
///
/// Two rules the log has to keep, both of them here rather than at each call
/// site. The write must not add its latency to a search or a read, so it runs
/// in its own task. And a write that fails must not fail the read it was
/// measuring, so its error becomes a log line and stops there.
///
/// The user id is captured before the task is spawned and re-installed inside
/// it. A `tokio::task_local` does not cross `tokio::spawn`, so a write that did
/// not carry it would run as the default user and scope itself to the wrong
/// rows.
///
/// `what` names the write in the log line. It is a static string so a failing
/// write is greppable without reading the arguments.
///
/// The caller's span crosses the spawn as well. A `tracing` span is task-local
/// like the user id, so without `in_current_span` the warning below would be
/// the one line of a turn that carries no correlation id - and it is a line an
/// operator only reads while looking for a turn.
pub fn record_in_background<F>(what: &'static str, write: F)
where
    F: Future<Output = Result<usize, CoreError>> + Send + 'static,
{
    use tracing::Instrument;
    let user_id = current_user_id();
    tokio::spawn(
        async move {
            match with_user_id(user_id, write).await {
                Ok(0) => {}
                Ok(rows) => tracing::debug!(target: "knowledge_use", what, rows, "use log written"),
                // A failing write is a fault, not an expected decline: an
                // unmigrated database, an exhausted pool or a missing grant makes
                // every write fail, and at debug the daemon would say nothing while
                // the tables stayed empty. Ranking would then score every entry
                // alike on the use terms - a ranking that looks like a working one,
                // which is the failure this log exists to prevent.
                Err(error) => tracing::warn!(
                    target: "knowledge_use",
                    what,
                    %error,
                    "use log write failed; the read it measured is unaffected"
                ),
            }
        }
        .in_current_span(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recall_offer_and_a_search_offer_are_different_scopes() {
        // The two are not interchangeable: one replaces the conversation's
        // standing offers and the other adds to them.
        assert_eq!(OfferScope::recall("c1").source, OfferSource::Recall);
        assert_eq!(OfferScope::search("c1").source, OfferSource::Search);
        assert_ne!(OfferScope::recall("c1"), OfferScope::search("c1"));
    }

    #[tokio::test]
    async fn a_background_write_that_fails_does_not_reach_the_caller() {
        // The call returns immediately and returns nothing to fail on. This is
        // the whole contract: a measurement cannot break what it measures.
        record_in_background("test", async {
            Err(CoreError::Storage("the database is gone".to_string()))
        });
    }
}
