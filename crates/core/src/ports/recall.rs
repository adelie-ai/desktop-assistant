//! Pre-prompt recall port (#1100): the lookup behind the `[Recall]` block.
//!
//! When a user prompt lands, the daemon embeds it once and asks two indexes
//! that share that embedding space - the knowledge base and the tag registry -
//! what is near it. The answer travels back through this port as candidates,
//! and [`crate::recall`] decides which of them clear the relevance floor and
//! how they render.
//!
//! ## Why the floor is not applied here
//!
//! The adapter owns the embedding call, the SQL, and the degradation to
//! full-text when the embedding is unavailable. The core owns the floor, the
//! caps, and the wording. Splitting it that way keeps every rule the block's
//! honesty rests on - what counts as relevant, how many lines fit, what the
//! "did not fit" count may claim - testable without a database.
//!
//! ## What the adapter owes this port
//!
//! - **Best match first.** Both lists arrive ordered, nearest first. The core
//!   never reorders them: it cannot compare a cosine distance with a lexical
//!   match, and it does not have to, because one lookup uses one mode.
//! - **One user.** Row-level security is a backstop the table owner bypasses,
//!   so every query behind this port carries its own `WHERE user_id`
//!   predicate.
//! - **No failure reaches the turn.** An adapter that cannot answer returns an
//!   error, and the caller drops the block and proceeds.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::KnowledgeEntry;

/// How near a candidate is to the prompt, and in which sense.
///
/// The two arms are not interchangeable, and the block's honesty depends on
/// keeping them apart: a cosine distance is a measured quantity that a floor
/// can be set against, while a full-text match is a yes/no answer the database
/// already made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecallRelevance {
    /// Cosine distance from the prompt embedding. pgvector's `<=>` returns a
    /// value in `[0, 2]`, and lower means nearer.
    Distance(f64),
    /// The row carries the prompt's search terms. Full-text match is binary -
    /// a row that does not match is never returned - so a row that arrives
    /// this way has already passed a floor of its own.
    ///
    /// This is what the arms degrade to when the embedding backend is
    /// unreachable or too slow (the precedent is #195).
    LexicalMatch,
}

impl RecallRelevance {
    /// Whether this candidate is near enough to the prompt to show.
    ///
    /// `max_distance` is the relevance floor stated as a cosine-distance
    /// ceiling: a candidate must sit at or under it. A [`Self::LexicalMatch`]
    /// always clears, because the database applied its own floor before the
    /// row travelled.
    pub fn clears_floor(self, max_distance: f64) -> bool {
        match self {
            Self::Distance(distance) => distance <= max_distance,
            Self::LexicalMatch => true,
        }
    }
}

/// One knowledge-base entry offered as a recall candidate.
///
/// The whole entry travels, not a pre-rendered line, so the block renders it
/// through [`KnowledgeEntry::display_line`] - the one place that decides what
/// stands for an entry that has no stored summary.
#[derive(Debug, Clone)]
pub struct RecallEntry {
    pub entry: KnowledgeEntry,
    pub relevance: RecallRelevance,
}

/// One tag name offered as a recall candidate.
///
/// The name alone: the point of the tag arm is a working vocabulary for the
/// model's first knowledge search, and a tag's description says what the tag
/// means rather than what this prompt is about.
#[derive(Debug, Clone)]
pub struct RecallTag {
    pub name: String,
    pub relevance: RecallRelevance,
}

/// What one recall lookup asks for.
///
/// Both limits are ceilings on rows *read*, not on rows shown. The block
/// renders far fewer, and reads further so that "and N more matched less
/// closely" is a number rather than a guess.
#[derive(Debug, Clone)]
pub struct RecallRequest {
    /// The user prompt, embedded once and asked of both indexes.
    pub prompt: String,
    /// Ceiling on knowledge rows read.
    pub entry_limit: usize,
    /// Ceiling on tag rows read.
    pub tag_limit: usize,
}

/// What one recall lookup found, each list nearest-first.
///
/// Empty lists are an ordinary answer: a prompt with nothing near it is the
/// case the relevance floor exists to keep quiet.
#[derive(Debug, Clone, Default)]
pub struct RecallCandidates {
    pub entries: Vec<RecallEntry>,
    pub tags: Vec<RecallTag>,
}

/// Boxed async closure that runs one recall lookup.
///
/// Wired by the daemon when a knowledge store is available and the feature is
/// enabled. Absent leaves the turn exactly as it was before the block existed.
pub type RecallSearchFn = Arc<
    dyn Fn(
            RecallRequest,
        ) -> Pin<Box<dyn Future<Output = Result<RecallCandidates, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_distance_at_the_floor_still_clears_it() {
        // The floor is a ceiling on distance, and the boundary belongs to the
        // side that shows the entry: a hit exactly at the tuned value is the
        // weakest one the tuning meant to keep.
        assert!(RecallRelevance::Distance(0.45).clears_floor(0.45));
        assert!(RecallRelevance::Distance(0.44).clears_floor(0.45));
        assert!(!RecallRelevance::Distance(0.46).clears_floor(0.45));
    }

    #[test]
    fn a_lexical_match_clears_any_floor() {
        // Full-text match is binary. A row that did not match is never
        // returned, so there is no distance to compare and nothing to drop.
        assert!(RecallRelevance::LexicalMatch.clears_floor(0.0));
        assert!(RecallRelevance::LexicalMatch.clears_floor(2.0));
    }
}
