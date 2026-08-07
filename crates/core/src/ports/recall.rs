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
//! - **Best match first.** Both lists arrive ordered, nearest first. That order
//!   is what the bar rests on - it drops a suffix only because a nearer row is
//!   never further down the list - and it is what a list of lexical matches is
//!   ranked by, because such a row carries no distance to score. The core
//!   reorders a list of measured candidates by activation once the bar has
//!   admitted them ([`crate::domain::activation`]), and never mixes the two
//!   kinds, because one lookup uses one mode.
//! - **What the use log knows, where it can be read.** A measured candidate
//!   carries its [`RecallEntry::use_record`], which is the reinforcement half of
//!   its activation. An adapter that cannot read the log answers `None` and the
//!   entry ranks on its semantic signal alone.
//! - **Each source's own dispersion, where it can measure one.** A distance
//!   means nothing until it is read against the spread of the source it came
//!   from - see [`RecallDispersion`]. The adapter measures that over the whole
//!   source, not over the rows it returned, and answers `None` when it cannot.
//! - **The present situation, read against the source.** A candidate carries
//!   the situations it has been seen in ([`RecallEntry::situation`]), and the
//!   answer carries one [`SituationCue`] for the whole lookup: the present
//!   situation, plus how much each of its values separates one entry of this
//!   store from another. That second half is a property of the source in the
//!   same way a dispersion is, so it is measured over the source and answered
//!   as `None` when it cannot be - see [`crate::domain::situation`].
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
use crate::domain::knowledge_use::KnowledgeUseRecord;
use crate::domain::situation::{SituationCue, SituationRecord};

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
    ///
    /// **It carries no semantic signal, so it is not ranked by activation**
    /// (#1123). There is no distance to read against the source's spread, so
    /// there is no dimensionless term for an activation score to add to. The
    /// two alternatives are both worse than leaving it alone:
    ///
    /// - Scoring it on the use log alone would order the block by what has been
    ///   opened most, discarding how well each row matched. The database
    ///   already ranked these rows by `ts_rank_cd`, which is a statement about
    ///   the match, and throwing that away would make an outage worse rather
    ///   than merely different.
    /// - Standing in a fixed semantic value for every row would say the rows are
    ///   equally good, which is the same loss written differently.
    ///
    /// So a lexical candidate keeps the order the database gave it. One lookup
    /// uses one mode, so a lexical candidate and a measured one are never in the
    /// same list and no mixed comparison arises - see the module header.
    LexicalMatch,
}

impl RecallRelevance {
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

    /// The dimensionless semantic term an activation score adds
    /// ([`crate::domain::activation`]), or `None` where this candidate has
    /// none.
    ///
    /// The same quantity [`Self::clears_bar`] compares, answered rather than
    /// tested, so the bar and the score read one number. `None` for a
    /// [`Self::LexicalMatch`], whose documentation says what follows from it.
    pub fn semantic_signal(self, dispersion: RecallDispersion) -> Option<f64> {
        match self {
            Self::Distance(distance) => Some(dispersion.deviations_below_median(distance)),
            Self::LexicalMatch => None,
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
    /// What the use log knows about this entry (#698), and the reinforcement
    /// half of its activation score (#1123).
    ///
    /// `None` is an ordinary answer with two causes the core does not have to
    /// tell apart: the log has never seen the entry, or the adapter could not
    /// read the log this turn. Either way the entry is ranked on its semantic
    /// signal alone, which is how every entry ranked before the log existed.
    pub use_record: Option<KnowledgeUseRecord>,
    /// The situations this entry has been seen in (#1125), and the third term
    /// of its activation score.
    ///
    /// Empty is an ordinary answer, with the same two causes and the same
    /// consequence as an absent `use_record`: an entry written before any of
    /// this was recorded, or an adapter that could not read the table. Either
    /// way [`SituationCue::coverage`] answers zero and the entry ranks the way
    /// it ranked before the cue existed.
    pub situation: SituationRecord,
}

impl RecallEntry {
    /// A candidate the use log has nothing to say about.
    ///
    /// The degraded read and the cold store both land here, and so does every
    /// test whose subject is not the use log.
    pub fn new(entry: KnowledgeEntry, relevance: RecallRelevance) -> Self {
        Self {
            entry,
            relevance,
            use_record: None,
            situation: SituationRecord::new(),
        }
    }

    /// The same candidate, carrying what the log knows about it.
    #[must_use]
    pub fn with_use_record(mut self, record: Option<KnowledgeUseRecord>) -> Self {
        self.use_record = record;
        self
    }

    /// The same candidate, carrying the situations it has been seen in.
    #[must_use]
    pub fn with_situation(mut self, situation: SituationRecord) -> Self {
        self.situation = situation;
        self
    }
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
    /// The knowledge source's own dispersion, measured over the whole source.
    /// `None` where the adapter could not measure one, which leaves the block on
    /// its stated estimate.
    pub entry_dispersion: Option<RecallDispersion>,
    /// The scratchpad source's own dispersion, on the same terms.
    pub note_dispersion: Option<RecallDispersion>,
    /// The present situation, read against the knowledge source (#1125).
    ///
    /// The same split as [`Self::entry_dispersion`], and for the same reason:
    /// how much a situation value separates one entry from another is a property
    /// of the whole source, so only the adapter can measure it, and a count
    /// taken over the candidates one lookup returned would describe the near
    /// tail instead. `None` where the adapter measured none - a store below its
    /// population floor, a deployment with nothing connected, or a read that
    /// failed - and every entry then ranks the way it ranked before the cue
    /// existed.
    pub situation_cue: Option<SituationCue>,
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
