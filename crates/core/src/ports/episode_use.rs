//! Outbound port for the episode use log (#698, #1350).
//!
//! The `[Recall]` block offers past turns the model never searched for, so the
//! same question the knowledge use log answers has to be answerable of an
//! episode: was it put in front of the model, and was it taken up? An episode
//! surfaced on twenty prompts and opened on none is an offer the arm would be
//! better without, and one opened again and again has earned its rank.
//!
//! **The arm ships with this and not after it.** An arm with no log ranks on
//! semantic distance alone, which is search rather than activation - and it can
//! never be judged, because nothing records whether an offered episode was ever
//! useful. The reinforcement half of the activation score
//! ([`crate::domain::activation`]) reads what comes back from
//! [`EpisodeUseLog::records`].
//!
//! ## Why not the knowledge use log, and why not the skill one
//!
//! Same shape, different key, on exactly the reasoning
//! [`crate::ports::skill_use`] gives. `knowledge_use_stats` and
//! `knowledge_offers` carry a foreign key to `knowledge_base(id)`, which frees
//! an entry's use rows when the entry is reaped; an episode has no row there.
//! `skill_use_stats` is keyed on a catalog name and deliberately carries no
//! foreign key, because a skill is never deleted - an episode is, whenever its
//! conversation is, so its use rows need a cascade of their own.
//!
//! What is shared is everything above the key. [`OfferScope`] decides whether
//! an offer replaces this conversation's standing set or adds to it, on the
//! rules [`crate::ports::knowledge_use`] states. The record that comes back is
//! a [`KnowledgeUseRecord`], because the reinforcement half of an activation
//! score is the same arithmetic over the same counters whatever was offered.
//! Its `entry_id` field carries the digest's row id, which is what names an
//! episode everywhere the model can reach one.
//!
//! ## No situation, and no marks
//!
//! Neither act has a writer, so neither has a table. No tool marks an episode
//! useful or wrong, and nothing records the situation an episode was opened in.
//! So [`EpisodeUseLog::record_opened`] takes no situation, and
//! [`crate::ports::recall::RecallEpisode`] answers the situation term with
//! `NO_SITUATION` rather than with a guess. Adding either act is a schema
//! change and a writer, together, not a column waiting for one.
//!
//! ## Recording never fails a read
//!
//! Every call site goes through
//! [`record_in_background`](crate::ports::knowledge_use::record_in_background),
//! for the reason that function states: a measurement must not be able to break
//! what it measures.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::knowledge_use::KnowledgeUseRecord;
use crate::ports::knowledge_use::OfferScope;

/// The episode use log: what was offered, and what was opened.
///
/// Every method is scoped to the current user through
/// [`crate::ports::auth::current_user_id`]. Counts returned are rows actually
/// written, so a caller can tell "recorded nothing" from "recorded everything
/// asked".
pub trait EpisodeUseLog: Send + Sync {
    /// Record that `episode_ids` were put in front of the model, and leave them
    /// standing as offers in `scope.conversation_id`.
    ///
    /// [`OfferSource::Recall`](crate::ports::knowledge_use::OfferSource::Recall)
    /// clears that conversation's standing episode offers first, because the
    /// block that produced it renders once per turn. An id the store does not
    /// hold for this person records nothing and is not an error.
    fn record_offered(
        &self,
        scope: OfferScope,
        episode_ids: Vec<String>,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// Record an open for each of `episode_ids` standing offered in
    /// `conversation_id`, and take those offers down.
    ///
    /// A read nothing offered records nothing: the model fetches an episode for
    /// many reasons, and only a taken-up offer is evidence that the block
    /// worked. Taking the offer down is also what makes the count idempotent,
    /// so a retried tool call adds nothing.
    fn record_opened(
        &self,
        conversation_id: String,
        episode_ids: Vec<String>,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// What the log knows about each of `episode_ids`.
    ///
    /// The read ranking uses. Ids with no record are absent from the answer
    /// rather than returned as zeroes, so a caller can tell an episode nothing
    /// has seen from one that was offered and ignored. Each record's `entry_id`
    /// carries the digest's row id.
    fn records(
        &self,
        episode_ids: Vec<String>,
    ) -> impl Future<Output = Result<Vec<KnowledgeUseRecord>, CoreError>> + Send;
}

/// Boxed async closure that records an episode offer, for wiring the log
/// through non-generic boundaries.
pub type EpisodeOfferedFn = Arc<
    dyn Fn(
            OfferScope,
            Vec<String>,
        ) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure that records episode opens against a conversation's
/// standing offers. Args: `(conversation_id, episode_ids)`.
pub type EpisodeOpenedFn = Arc<
    dyn Fn(String, Vec<String>) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;
