//! Pre-prompt recall port (#1100, #1101): the lookup behind the `[Recall]`
//! block.
//!
//! When a user prompt lands, the daemon embeds it once and asks the two indexes
//! that share that embedding space - the knowledge base and this conversation's
//! scratchpad - what is near it. The answer travels back through this port as
//! candidates, and [`crate::recall`] decides which of them clear the bar and how
//! they render.
//!
//! ## Why the bar is not applied here
//!
//! The adapter owns the embedding call, the SQL, and the degradation to
//! full-text when the embedding is unavailable. The core owns the bar, the caps,
//! and the wording. Splitting it that way keeps every rule the block's honesty
//! rests on - what counts as relevant, how many lines fit, what the "did not
//! fit" count may claim - testable without a database.
//!
//! ## What the adapter owes this port
//!
//! - **Best match first.** Both lists arrive ordered, nearest first. The core
//!   never reorders them: it cannot compare a cosine distance with a lexical
//!   match, and it does not have to, because one lookup uses one mode.
//! - **Each source's own dispersion, where it can measure one.** A distance
//!   means nothing until it is read against the spread of the source it came
//!   from - see [`RecallDispersion`]. The adapter measures that over the whole
//!   source, not over the rows it returned, and answers `None` when it cannot.
//! - **One user, and one conversation's pad.** Row-level security is a backstop
//!   the table owner bypasses, so every query behind this port carries its own
//!   `WHERE user_id` predicate. The scratchpad arm carries a `conversation_id`
//!   predicate beside it: the pad is per-conversation by design, and reaching
//!   across conversations is a different feature with its own privacy question.
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

    /// Whether this candidate stands out from its source far enough to show.
    ///
    /// `bar` is dimensionless: how many median absolute deviations below the
    /// source's own median a candidate must sit - see
    /// [`RecallDispersion::deviations_below_median`]. A [`Self::LexicalMatch`]
    /// always clears, because the database applied its own floor before the row
    /// travelled and there is no distance to read.
    pub fn clears_bar(self, dispersion: RecallDispersion, bar: f64) -> bool {
        match self {
            Self::Distance(distance) => dispersion.deviations_below_median(distance) >= bar,
            Self::LexicalMatch => true,
        }
    }
}

/// How spread out one source's distances are, so a distance from that source
/// can be read in the source's own terms.
///
/// A cosine distance carries no meaning on its own. What counts as near depends
/// on the embedding model, on how much text a row holds, and on how wide the
/// subject matter of the store is, so a number fitted to one deployment says
/// nothing about the next one. What does carry is the shape of the
/// distribution: over any store, a query with a real cue produces a few rows far
/// below the store's usual distance, and a query with no cue produces none.
///
/// The median and the median absolute deviation state that shape. Both are
/// robust to the near tail - which is exactly the part a cued prompt moves - so
/// they describe the store's ordinary geometry rather than the answer to one
/// query.
///
/// **Measured over the source, never over the candidates.** A lookup reads only
/// the nearest rows, so their spread is the near tail's and not the source's,
/// and normalizing inside it inflates every score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallDispersion {
    /// The distance a middling row of this source sits at.
    median: f64,
    /// The median absolute deviation of those distances: the unit the bar is
    /// stated in.
    deviation: f64,
}

/// How many rows a dispersion must be measured over before it is used.
///
/// A median absolute deviation over a handful of rows is noise, and a noisy
/// unit makes a dimensionless bar meaningless. A source below this count has no
/// measurable geometry yet, and the caller falls back to a stated estimate.
pub const RECALL_DISPERSION_MIN_ROWS: usize = 20;

/// The narrowest spread a measurement may claim, as a fraction of its own
/// median.
///
/// A degenerate store - one where most rows carry near-identical text - reports
/// a deviation close to zero, and dividing by it puts every row a shade nearer
/// than the median far past any bar. That would render half the store on every
/// prompt. The rule is a ratio rather than a distance so that it carries across
/// models the same way the bar does.
pub const RECALL_DISPERSION_MIN_RELATIVE_SPREAD: f64 = 0.02;

impl RecallDispersion {
    /// A stated estimate, for a source whose own geometry is not known yet.
    ///
    /// Const so a fallback can be a constant. Nothing is checked here, because
    /// the values are the caller's own and not a measurement - see
    /// [`Self::measured`] for the checked path.
    pub const fn assumed(median: f64, deviation: f64) -> Self {
        Self { median, deviation }
    }

    /// A measurement of one source, or `None` where it cannot be trusted.
    ///
    /// `rows` is how many rows the two statistics were measured over. The
    /// answer is `None` for a sample under [`RECALL_DISPERSION_MIN_ROWS`], for a
    /// value that is not finite, and for a spread under
    /// [`RECALL_DISPERSION_MIN_RELATIVE_SPREAD`] of the median. Every one of
    /// those leaves the caller on its stated estimate, which is the quiet
    /// answer rather than the loud one.
    pub fn measured(median: f64, deviation: f64, rows: usize) -> Option<Self> {
        if rows < RECALL_DISPERSION_MIN_ROWS
            || !median.is_finite()
            || !deviation.is_finite()
            || median <= 0.0
            || deviation < median * RECALL_DISPERSION_MIN_RELATIVE_SPREAD
        {
            return None;
        }
        Some(Self::assumed(median, deviation))
    }

    /// How far below this source's median `distance` sits, counted in the
    /// source's own median absolute deviations.
    ///
    /// Higher is nearer. The quantity is dimensionless, so one bar reads the
    /// same against any source and any embedding model.
    pub fn deviations_below_median(self, distance: f64) -> f64 {
        (self.median - distance) / self.deviation
    }

    /// The distance a candidate that stands `deviations` out of this source
    /// sits at: the inverse of [`Self::deviations_below_median`].
    ///
    /// It states a position in one source's geometry as the distance that
    /// source would report, so a fixed set of positions describes any source.
    pub fn distance_at(self, deviations: f64) -> f64 {
        self.median - deviations * self.deviation
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

/// One scratchpad note offered as a recall candidate (#1101).
///
/// The note itself travels, not a pre-rendered line, because how much of a note
/// a line may spend and what counts as a note already in view are the core's
/// decisions - see [`crate::recall`].
#[derive(Debug, Clone)]
pub struct RecallNote {
    /// The note's key, as the model wrote it. Nothing between the write tool
    /// and here bounds its length or its characters.
    pub key: String,
    /// The note's content, up to
    /// [`MAX_NOTE_BYTES`](crate::ports::scratchpad::MAX_NOTE_BYTES).
    pub content: String,
    /// Whether the note is pinned, so `[Pinned]` already carries its whole
    /// content this turn. The flag travels rather than the adapter filtering on
    /// it, because "already in view" is one rule and it is applied in one place.
    pub pinned: bool,
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
/// Every limit is a ceiling on rows *read*, not on rows shown. The block
/// renders far fewer, and reads further so that "and N more matched less
/// closely" is a number rather than a guess.
#[derive(Debug, Clone)]
pub struct RecallRequest {
    /// The user prompt, embedded once and asked of every index.
    pub prompt: String,
    /// The conversation whose scratchpad the note arm reads. The pad is
    /// per-conversation, so this is a scope and not a hint.
    pub conversation_id: String,
    /// Ceiling on knowledge rows read.
    pub entry_limit: usize,
    /// Ceiling on scratchpad rows read.
    pub note_limit: usize,
    /// Ceiling on tag rows read.
    pub tag_limit: usize,
}

/// What one recall lookup found, each list nearest-first, and how spread out
/// the source each list came from is.
///
/// Empty lists are an ordinary answer: a prompt with nothing near it is the case
/// the bar exists to keep quiet. So is one arm empty and the others full - an
/// arm that could not answer contributes nothing and the rest of the block still
/// renders.
#[derive(Debug, Clone, Default)]
pub struct RecallCandidates {
    pub entries: Vec<RecallEntry>,
    pub notes: Vec<RecallNote>,
    pub tags: Vec<RecallTag>,
    /// The knowledge source's own dispersion, measured over the whole source.
    /// `None` where the adapter could not measure one, which leaves the block on
    /// its stated estimate.
    pub entry_dispersion: Option<RecallDispersion>,
    /// The scratchpad source's own dispersion, on the same terms.
    pub note_dispersion: Option<RecallDispersion>,
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

    /// A source whose middling row sits at 0.80 and whose distances vary by
    /// 0.05. One deviation is then 0.05 of cosine distance.
    fn a_source() -> RecallDispersion {
        RecallDispersion::assumed(0.80, 0.05)
    }

    #[test]
    fn a_candidate_at_the_bar_still_clears_it() {
        // The bar is a floor on how far a candidate stands out, and the
        // boundary belongs to the side that shows the entry: a candidate
        // exactly at the bar is the weakest one the bar meant to keep.
        let source = a_source();
        assert!(RecallRelevance::Distance(0.50).clears_bar(source, 6.0));
        assert!(RecallRelevance::Distance(0.49).clears_bar(source, 6.0));
        assert!(!RecallRelevance::Distance(0.51).clears_bar(source, 6.0));
    }

    #[test]
    fn the_same_distance_reads_differently_against_two_sources() {
        // The whole point of the unit. 0.50 is six deviations out of a source
        // whose rows vary by 0.05, and one deviation out of a source whose rows
        // vary by 0.30 - so it is exceptional in the first and ordinary in the
        // second, and no distance decides that on its own.
        let tight = RecallDispersion::assumed(0.80, 0.05);
        let loose = RecallDispersion::assumed(0.80, 0.30);

        assert!(RecallRelevance::Distance(0.50).clears_bar(tight, 6.0));
        assert!(!RecallRelevance::Distance(0.50).clears_bar(loose, 6.0));
    }

    #[test]
    fn a_lexical_match_clears_any_bar() {
        // Full-text match is binary. A row that did not match is never
        // returned, so there is no distance to compare and nothing to drop.
        assert!(RecallRelevance::LexicalMatch.clears_bar(a_source(), 0.0));
        assert!(RecallRelevance::LexicalMatch.clears_bar(a_source(), 1_000.0));
    }

    #[test]
    fn a_dispersion_measured_over_too_few_rows_is_refused() {
        // A median absolute deviation over a handful of rows is noise, and a
        // noisy unit makes the bar meaningless.
        assert_eq!(
            RecallDispersion::measured(0.80, 0.05, RECALL_DISPERSION_MIN_ROWS - 1),
            None
        );
        assert_eq!(
            RecallDispersion::measured(0.80, 0.05, RECALL_DISPERSION_MIN_ROWS),
            Some(RecallDispersion::assumed(0.80, 0.05))
        );
    }

    #[test]
    fn a_degenerate_spread_is_refused_rather_than_divided_by() {
        // A store of near-identical rows reports almost no spread, and dividing
        // by it would put every row a shade nearer than the median past any
        // bar - which renders half the store on every prompt.
        let rows = RECALL_DISPERSION_MIN_ROWS;
        assert_eq!(RecallDispersion::measured(0.80, 0.0, rows), None);
        assert_eq!(
            RecallDispersion::measured(
                0.80,
                0.80 * RECALL_DISPERSION_MIN_RELATIVE_SPREAD / 2.0,
                rows
            ),
            None
        );
        assert!(
            RecallDispersion::measured(0.80, 0.80 * RECALL_DISPERSION_MIN_RELATIVE_SPREAD, rows)
                .is_some()
        );
    }

    #[test]
    fn a_measurement_that_is_not_a_number_is_refused() {
        let rows = RECALL_DISPERSION_MIN_ROWS;
        assert_eq!(RecallDispersion::measured(f64::NAN, 0.05, rows), None);
        assert_eq!(RecallDispersion::measured(0.80, f64::NAN, rows), None);
        assert_eq!(RecallDispersion::measured(f64::INFINITY, 0.05, rows), None);
        assert_eq!(RecallDispersion::measured(0.0, 0.05, rows), None);
    }

    #[test]
    fn the_score_counts_deviations_below_the_median() {
        let source = a_source();
        assert!((source.deviations_below_median(0.80) - 0.0).abs() < 1e-9);
        assert!((source.deviations_below_median(0.55) - 5.0).abs() < 1e-9);
        // A row further out than the median scores negative, which no bar
        // admits.
        assert!(source.deviations_below_median(0.90) < 0.0);
    }
}
