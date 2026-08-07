//! Outbound port for the skill use log (#1154).
//!
//! The `[Recall]` block offers skills the model never searched for, so the same
//! question the knowledge use log answers now has to be answerable of a
//! procedure: was it put in front of the model, and was it taken up? A skill
//! surfaced on twenty prompts and opened on none is a skill the catalog would
//! be better without, and a skill opened again and again has earned its rank.
//! Neither fact exists unless something records it.
//!
//! ## Why not the knowledge use log
//!
//! Same shape, different key, and the key is not negotiable.
//! `knowledge_use_stats` and `knowledge_offers` carry a foreign key to
//! `knowledge_base(id)`, which is what frees an entry's use rows when the entry
//! is reaped. A skill has no row in that table, so it cannot be recorded there
//! without dropping the key that makes the knowledge log correct.
//!
//! What is shared is everything above the key. [`OfferScope`] decides whether an
//! offer replaces this conversation's standing set or adds to it, on exactly the
//! rules the knowledge log's module documentation states. The record that comes
//! back is a [`KnowledgeUseRecord`], because the reinforcement half of an
//! activation score is the same arithmetic over the same counters whatever was
//! offered - see [`crate::domain::activation`]. Its `entry_id` field carries the
//! skill's name, which is what names a skill everywhere else: it is the handle
//! `builtin_skill_get` takes.
//!
//! ## What a skill is keyed by
//!
//! Its name, scoped to the reading user. The catalog is host-global with a
//! per-row `owner_user_id`, and a reader sees the global skills plus their own,
//! so a name resolves to exactly one skill for one reader. The log is
//! per-user because the use is: one person's opens say nothing about another's.
//!
//! ## Marks have no writer yet
//!
//! The knowledge log records a third act - a mark, "this helped" or "this was
//! wrong". No tool sets one on a skill, so this port has no method for it and
//! the schema behind it has no table. A record therefore comes back with no
//! marks, and the reinforcement term reads offers and opens alone. Adding the
//! third act is a schema change and a tool, together, not a column waiting for
//! one.
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

/// The skill use log: what was offered, and what was opened.
///
/// Every method is scoped to the current user through
/// [`crate::ports::auth::current_user_id`]. Counts returned are rows actually
/// written, so a caller can tell "recorded nothing" from "recorded everything
/// asked".
pub trait SkillUseLog: Send + Sync {
    /// Record that `names` were put in front of the model, and leave them
    /// standing as offers in `scope.conversation_id`.
    ///
    /// [`OfferSource::Recall`](crate::ports::knowledge_use::OfferSource::Recall)
    /// clears that conversation's standing skill offers first, because the block
    /// that produced it renders once per turn. A name the catalog does not hold
    /// records nothing and is not an error.
    fn record_offered(
        &self,
        scope: OfferScope,
        names: Vec<String>,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// Record an open for each of `names` standing offered in
    /// `conversation_id`, and take those offers down.
    ///
    /// A read nothing offered records nothing: the model opens a skill for many
    /// reasons, and only a taken-up offer is evidence that the block worked.
    /// Taking the offer down is also what makes the count idempotent, so a
    /// retried tool call adds nothing.
    fn record_opened(
        &self,
        conversation_id: String,
        names: Vec<String>,
    ) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// What the log knows about each of `names`.
    ///
    /// The read ranking uses. Names with no record are absent from the answer
    /// rather than returned as zeroes, so a caller can tell a skill nothing has
    /// seen from one that was offered and ignored. Each record's `entry_id`
    /// carries the skill's name.
    fn records(
        &self,
        names: Vec<String>,
    ) -> impl Future<Output = Result<Vec<KnowledgeUseRecord>, CoreError>> + Send;
}

/// Boxed async closure that records a skill offer, for wiring the log through
/// non-generic boundaries.
pub type SkillOfferedFn = Arc<
    dyn Fn(
            OfferScope,
            Vec<String>,
        ) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure that records skill opens against a conversation's
/// standing offers. Args: `(conversation_id, skill_names)`.
pub type SkillOpenedFn = Arc<
    dyn Fn(String, Vec<String>) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;
