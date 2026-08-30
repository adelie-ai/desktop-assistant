//! One in-memory record per turn of what the `[Recall]` lookup considered
//! (#1327): every candidate the scan returned, each one's activation score
//! broken down by term, which candidates were offered, and which were later
//! opened.
//!
//! ## The question this answers, and why a record was needed for it
//!
//! [`crate::domain::activation`] scores every admitted candidate on every
//! turn and [`crate::ports::recall::rank_by_activation`] keeps only the
//! order. When retrieval puts the right answer out of reach, nothing says
//! why: not which candidates were even in play, and not how each one scored
//! by term. Reconstructing that by hand means re-embedding the prompt,
//! ranking the corpus again, and hoping the rebuild used the same weights the
//! turn did.
//!
//! This type is what `render_recall` builds instead, from
//! the same ranking pass that renders the block - see
//! [`crate::ports::recall::rank_by_activation_traced`]. The score a reader
//! sees here is the score the turn ranked on, because it is the same
//! computation and not a second one.
//!
//! ## Scope
//!
//! This is not a general telemetry system. It answers two questions in
//! flight: whether a confidence term improves the rank of a known answer, and
//! whether a change to extraction reduces false claims. A candidate's score,
//! whether it cleared the bar, whether it rendered, and why not, are what
//! those questions need. Nothing here aggregates across turns; a reader who
//! wants that runs the replay harness against a snapshot instead.
//!
//! ## Persistence is a later unit
//!
//! This module holds the in-memory shape only. [`ContextPlanRecordFn`] and
//! [`ContextPlanOpenedFn`] are the boxed closures a store wires in, mirroring
//! [`ContextBreakdownRecordFn`](crate::ports::context_breakdown::ContextBreakdownRecordFn):
//! the turn loop holds one of these rather than a store, so `core` depends on
//! no storage crate, and a deployment with no database records nothing
//! instead of failing turns.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::activation::{ActivationTerms, ActivationWeights};
use crate::domain::knowledge_use::KnowledgeUseRecord;
use crate::ports::recall::{RecallDispersion, RecallRelevance};

/// The most candidates one plan keeps, however many the lookup considered.
///
/// A turn's scan limits cap what today's arms can return well below this -
/// [`MAX_PLANNED_CANDIDATES`] is the plan's own ceiling, not theirs, so a
/// future scan limit can grow without the stored array silently growing
/// unbounded alongside it. [`ContextPlan::considered_count`] keeps the true
/// count whether or not the array itself was cut, so a cut plan never reads
/// as a small turn.
///
/// **This bounds what is kept, not what is scored.** The plan builder scores
/// every candidate the scan returned before this cap runs; the cap only
/// truncates the finished [`Vec`] afterward. A future scan limit large enough
/// to matter would still pay to score the full set - this constant protects
/// the size of the record, not the cost of building it.
pub const MAX_PLANNED_CANDIDATES: usize = 512;

/// The most bytes of the prompt's own query text a plan keeps.
///
/// The text is what turns a failing turn into a labelled case with one
/// command (#1328), not a transcript - a query is a sentence or two, and
/// anything past this bound is kept truncated rather than dropped, so the
/// case can still be seeded from the words that are there.
pub const MAX_QUERY_TEXT_BYTES: usize = 8 * 1024;

/// Which arm of the lookup a candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecallArm {
    /// The durable knowledge base.
    Entry,
    /// This conversation's scratchpad.
    Note,
    /// The skill catalog.
    Skill,
    /// The episodic turn index (#1350).
    Episode,
}

/// Why an admitted candidate did not render, when it did not.
///
/// Only an admitted candidate (`cleared_bar: true`) carries one of these; a
/// candidate the bar itself refused needs no further reason; see
/// [`PlannedCandidate::cleared_bar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannedDropReason {
    /// The block's width ran out before this candidate's line - it ranked
    /// below the last one the budget had room for.
    WidthCap,
    /// The candidate is already carried in full elsewhere in the prompt:
    /// under `[Pinned]`, or a scratchpad note the model pinned itself.
    Pinned,
    /// The candidate is already listed elsewhere in the prompt: a key the
    /// `[Scratchpad]` index already shows, or a step or finding `[Plan]` has
    /// already named.
    InView,
    /// The candidate's id or name does not survive the block's bound, so a
    /// fetch by the id the model would be shown could never resolve it.
    IdUnrenderable,
    /// The candidate's rendered line came out empty once its content was
    /// read - nothing to show, so nothing was.
    EmptyContent,
    /// The candidate carries content from outside the trust boundary that
    /// this turn withholds from the model (#1247), or is tainted by a marker
    /// this block cannot fold into the turn's own provenance.
    ExternalContent,
    /// The candidate's line was near enough to a better-ranked one that the
    /// two render as one line and a count (#1350). It is shown, on somebody
    /// else's line, rather than absent - which is a different fact about the
    /// turn from running out of width, so the plan states it separately.
    NearDuplicate,
}

/// What the use log knew about a candidate at score time (#698), so
/// "reinforcement contributed nothing" is checkable against its own input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedUseCounts {
    /// Times this candidate has been offered before.
    pub offered: u64,
    /// Times a fetch actually opened it.
    pub opened: u64,
    /// Times a person or the model marked it right or wrong.
    pub marked: u64,
}

impl PlannedUseCounts {
    /// Read off a [`KnowledgeUseRecord`], the same counters the reinforcement
    /// term was computed from.
    pub fn from_record(record: &KnowledgeUseRecord) -> Self {
        Self {
            offered: record.offered_count,
            opened: record.opened_count,
            marked: record.marked_count,
        }
    }
}

/// One candidate the lookup considered, whether or not it was offered.
///
/// "Considered" means the scan returned it - every row up to the arm's own
/// scan limit, not only the ones that cleared the bar. A row past the scan
/// limit was never considered, and [`ArmSummary::capped`] says so rather than
/// this type pretending to know about it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedCandidate {
    /// Which arm this candidate came from.
    pub arm: RecallArm,
    /// The entry id, note key, or skill name, exactly as the source holds
    /// it - not the bounded form the block may render.
    pub id: String,
    /// The raw relevance the adapter delivered: a cosine distance or a
    /// full-text match, tagged which.
    pub relevance: RecallRelevance,
    /// This candidate's activation score, broken down by term.
    ///
    /// Computed by the same trace function every candidate in this plan is
    /// scored by, whether or not the candidate cleared the bar - see
    /// [`crate::ports::recall::rank_by_activation_traced`].
    pub terms: ActivationTerms,
    /// What the use log knew about this candidate, or `None` where the
    /// source carries no use record for it (every scratchpad note, and an
    /// entry or skill the log has never seen).
    pub use_counts: Option<PlannedUseCounts>,
    /// Whether this candidate stood far enough out of its source to clear
    /// [`crate::recall::RECALL_BAR`].
    pub cleared_bar: bool,
    /// Position after activation ranking, best first, among this arm's
    /// admitted candidates. `None` for a candidate the bar refused, and also
    /// for a scratchpad note: the pad arm is never reordered by activation,
    /// so no ranking pass ever placed one.
    pub rank: Option<usize>,
    /// Whether this candidate's line actually rendered in the block.
    pub offered: bool,
    /// Why an admitted candidate did not render, when it did not. Always
    /// `None` when [`Self::offered`] is `true`, and always `None` when
    /// [`Self::cleared_bar`] is `false` - a candidate the bar refused needs
    /// no further reason.
    pub drop_reason: Option<PlannedDropReason>,
}

/// How one arm was scanned, and what its geometry looked like this turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmSummary {
    /// The dispersion this arm's candidates were read against to compute
    /// [`ActivationTerms::semantic`]: the source's own measurement wherever
    /// the source could supply one, and the stated estimate otherwise. See
    /// [`Self::dispersion_measured`] for which.
    pub dispersion: RecallDispersion,
    /// Whether [`Self::dispersion`] came from the source's own measurement
    /// (`true`) or from the stated estimate the block falls back to when the
    /// source has none (`false`).
    pub dispersion_measured: bool,
    /// Whether a situation cue was available for this arm this turn.
    pub situation_cue_present: bool,
    /// The ceiling this arm's scan was asked to read to.
    pub scan_limit: usize,
    /// How many rows this arm's scan actually returned.
    pub rows_returned: usize,
    /// Whether the scan filled up to [`Self::scan_limit`] with rows that all
    /// cleared the bar - the same hedge `render_recall`
    /// reports in its own "and N more" line, so `considered_count` on a
    /// capped turn is a lower bound rather than an exact count.
    pub capped: bool,
}

impl ArmSummary {
    /// An arm the lookup never reached: no rows, nothing measured, nothing
    /// capped. What every arm reads as on a turn that ran no lookup at all.
    pub fn empty(scan_limit: usize) -> Self {
        Self {
            dispersion: RecallDispersion::assumed(0.0, 1.0),
            dispersion_measured: false,
            situation_cue_present: false,
            scan_limit,
            rows_returned: 0,
            capped: false,
        }
    }
}

/// The four arms' summaries: entry, note, skill, episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmSummaries {
    pub entries: ArmSummary,
    pub notes: ArmSummary,
    pub skills: ArmSummary,
    /// The episodic turn index (#1350).
    pub episodes: ArmSummary,
}

/// One turn's context plan (#1327): what the `[Recall]` lookup considered,
/// how each candidate scored, and which it offered.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPlan {
    /// The turn's correlation id. Client-chosen, so a store scopes reads of
    /// it by user (the #1252 lesson).
    pub request_id: String,
    /// The conversation the turn ran in.
    pub conversation_id: String,
    /// Whether a recall lookup ran at all this turn. `false` for a turn whose
    /// first round rendered no recall surface (the feature is unwired, or the
    /// turn carried no anchor to lookup against) - the row still exists: "no
    /// retrieval" is a record, not an absence.
    pub recall_ran: bool,
    /// The prompt text the lookup embedded, bounded to
    /// [`MAX_QUERY_TEXT_BYTES`]. `None` when [`Self::recall_ran`] is `false`.
    pub query_text: Option<String>,
    /// Whether [`Self::query_text`] was cut to fit the bound.
    pub query_text_truncated: bool,
    /// [`crate::recall::RECALL_BAR`] as applied this turn.
    pub bar: f64,
    /// The [`ActivationWeights`] every candidate in [`Self::candidates`] was
    /// scored under.
    pub weights: ActivationWeights,
    /// Which shape of [`ActivationTerms`] [`Self::weights`] produced - see
    /// [`crate::domain::activation::ACTIVATION_SCORER_VERSION`].
    pub scorer_version: String,
    /// The three arms' scan summaries.
    pub arms: ArmSummaries,
    /// Every candidate the lookup considered, in ranked order within each
    /// arm - admitted candidates first, best first, then the candidates the
    /// bar refused. Cut to [`MAX_PLANNED_CANDIDATES`]; see
    /// [`Self::truncated`].
    pub candidates: Vec<PlannedCandidate>,
    /// The true number of candidates the lookup considered, before
    /// [`Self::candidates`] was cut to its cap. Equal to
    /// `candidates.len()` unless [`Self::truncated`] is set.
    pub considered_count: usize,
    /// Whether [`Self::candidates`] was cut to [`MAX_PLANNED_CANDIDATES`].
    /// [`Self::considered_count`] keeps the true count either way, so a cut
    /// plan never reads as a smaller turn than it was.
    pub truncated: bool,
    /// Ids fetched by the model during this turn, in the order they were
    /// opened. Appended after the plan's first write - see the module
    /// header.
    pub opened: Vec<String>,
    /// When the plan was recorded, RFC3339. `None` on a plan that has not
    /// been stored yet - the store assigns it.
    pub recorded_at: Option<String>,
}

impl ContextPlan {
    /// A turn whose first round rendered no recall surface at all: no anchor
    /// to look up against, or the feature unwired. `recall_ran` is `false`
    /// and every array is empty - the row still exists (#1327).
    pub fn no_lookup(request_id: impl Into<String>, conversation_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            conversation_id: conversation_id.into(),
            recall_ran: false,
            query_text: None,
            query_text_truncated: false,
            bar: 0.0,
            weights: ActivationWeights::default(),
            scorer_version: crate::domain::activation::ACTIVATION_SCORER_VERSION.to_string(),
            arms: ArmSummaries {
                entries: ArmSummary::empty(0),
                notes: ArmSummary::empty(0),
                skills: ArmSummary::empty(0),
                episodes: ArmSummary::empty(0),
            },
            candidates: Vec::new(),
            considered_count: 0,
            truncated: false,
            opened: Vec::new(),
            recorded_at: None,
        }
    }

    /// Stamp the turn identity onto a plan the recall lookup built, which
    /// knows what it considered but not which turn asked for it.
    ///
    /// `query_text` is bounded to [`MAX_QUERY_TEXT_BYTES`], on a character
    /// boundary, with [`Self::query_text_truncated`] set when it was cut -
    /// the same shape [`crate::ports::context_breakdown`] states for its own
    /// bounded fields.
    #[must_use]
    pub fn identify(
        mut self,
        request_id: impl Into<String>,
        conversation_id: impl Into<String>,
        query_text: &str,
    ) -> Self {
        self.request_id = request_id.into();
        self.conversation_id = conversation_id.into();
        let (bounded, truncated) = bound_query_text(query_text);
        self.query_text = Some(bounded);
        self.query_text_truncated = truncated;
        self
    }
}

/// Cut `text` to [`MAX_QUERY_TEXT_BYTES`] on a character boundary, and say
/// whether it was cut.
fn bound_query_text(text: &str) -> (String, bool) {
    if text.len() <= MAX_QUERY_TEXT_BYTES {
        return (text.to_string(), false);
    }
    let mut cut = MAX_QUERY_TEXT_BYTES;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    (text[..cut].to_string(), true)
}

/// Boxed async closure for the write: one plan, once per turn's first round.
///
/// The turn loop holds one of these rather than a store, so `core` records
/// without depending on any storage crate, and a deployment with no database
/// records nothing instead of failing turns. A failing write is logged and
/// swallowed - the same rule `persist_context_breakdown` follows for
/// [`crate::ports::context_breakdown::ContextBreakdownRecordFn`].
pub type ContextPlanRecordFn = Arc<
    dyn Fn(ContextPlan) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for the opened-append: `(request_id, opened_id)`.
///
/// Called once per id the model fetches during a turn, so a plan gains its
/// `opened` entries after the initial write rather than waiting for the turn
/// to finish - see the module header.
pub type ContextPlanOpenedFn = Arc<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::knowledge_use::{KnowledgeMark, MarkPolarity, MarkSource};
    use chrono::Utc;

    fn use_record() -> KnowledgeUseRecord {
        KnowledgeUseRecord {
            entry_id: "kb-1".to_string(),
            offered_count: 4,
            opened_count: 2,
            marked_count: 1,
            first_seen_at: Utc::now(),
            last_offered_at: Some(Utc::now()),
            recent_uses: Vec::new(),
            marks: vec![KnowledgeMark {
                source: MarkSource::Person,
                polarity: MarkPolarity::Positive,
                reason: None,
                marked_at: Utc::now(),
            }],
        }
    }

    #[test]
    fn planned_use_counts_read_the_same_counters_the_reinforcement_term_reads() {
        let record = use_record();
        let counts = PlannedUseCounts::from_record(&record);
        assert_eq!(counts.offered, 4);
        assert_eq!(counts.opened, 2);
        assert_eq!(counts.marked, 1);
    }

    #[test]
    fn a_turn_with_no_lookup_is_recorded_rather_than_absent() {
        let plan = ContextPlan::no_lookup("r1", "c1");
        assert!(!plan.recall_ran);
        assert!(plan.candidates.is_empty());
        assert_eq!(plan.considered_count, 0);
        assert!(!plan.truncated);
        assert!(plan.query_text.is_none());
    }

    // The distinction between "offered" and "opened" is covered where it can
    // actually be driven: `offered` through `render_recall_with_width` in
    // `crate::recall`'s test module (a candidate that clears the bar but
    // does not fit the render width, versus one that does), and `opened`
    // by the unit that wires the writer - nothing in this crate populates
    // `ContextPlan::opened` yet, so a test of it here would assert a
    // distinction the code cannot make.
}
