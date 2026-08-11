//! Pre-prompt recall port (#1100, #1101, #1154): the lookup behind the
//! `[Recall]` block.
//!
//! When a user prompt lands, the daemon embeds it once and asks every index
//! that shares that embedding space - the knowledge base, this conversation's
//! scratchpad, and the skill catalog - what is near it. The answer travels back
//! through this port as candidates, and [`crate::recall`] decides which of them
//! clear the bar and how they render.
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
//! - **Best match first.** Every list arrives ordered, nearest first. That order
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
//!   The skill arm reads a host-global catalog, so its scope predicate is the
//!   catalog's own: the global skills, plus this user's.
//! - **Only a skill a person approved** (#1154, #1155). Approval is consent -
//!   whether somebody agreed the procedure may be followed - and it is a
//!   separate axis from provenance. The adapter applies it, so an unapproved
//!   skill reaches neither the candidates nor the spread they are read against;
//!   [`RecallSkill`] says why that is an exclusion rather than a mark.
//! - **No failure reaches the turn.** An adapter that cannot answer returns an
//!   error, and the caller drops the block and proceeds.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::KnowledgeEntry;
use crate::domain::activation::{NO_SALIENCE, NO_SITUATION};
use crate::domain::knowledge_use::KnowledgeUseRecord;
use crate::domain::salience::{SalienceReading, SalienceSource};
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

/// The widest spread a measurement may claim, as a fraction of its own median.
///
/// The mirror of [`RECALL_DISPERSION_MIN_RELATIVE_SPREAD`], and refused for the
/// same reason: a measurement that cannot separate one row from another is not
/// a calibration. Too narrow a spread admits the whole source; too wide a one
/// admits nothing at all, including an exact match.
///
/// It is the bar's own reciprocal rather than a chosen number. The bar is
/// stated in deviations, so a source whose deviation exceeds `median / bar` has
/// `distance_at(bar) <= 0` - and a cosine distance is never negative, so no row
/// can clear however near it is. A source measured that way has not become
/// strict, it has become unreadable.
///
/// **Measured, not assumed** (#1243). A knowledge store measures a deviation
/// near a twenty-fifth of its median and never approaches this, so the guard
/// costs the store nothing. A single-task scratchpad measures near a sixth and
/// crosses it on about half of on-subject prompts - which is the case that
/// found this: on such a prompt the pad refused every note including a perfect
/// match, while an *unrelated* prompt, whose distances group tightly far away,
/// admitted notes freely. The arm was inverted on exactly the prompts it exists
/// to serve, and a bar out of reach is what did it.
pub const RECALL_DISPERSION_MAX_RELATIVE_SPREAD: f64 = 1.0 / crate::recall::RECALL_BAR;

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
    /// value that is not finite, for a spread under
    /// [`RECALL_DISPERSION_MIN_RELATIVE_SPREAD`] of the median, and for one over
    /// [`RECALL_DISPERSION_MAX_RELATIVE_SPREAD`] of it. Every one of those
    /// leaves the caller on its stated estimate, which is the quiet answer
    /// rather than the loud one.
    ///
    /// The two spread guards are one rule read from both ends: a measurement
    /// that cannot separate one row from another is refused whether it would
    /// admit everything or nothing.
    pub fn measured(median: f64, deviation: f64, rows: usize) -> Option<Self> {
        if rows < RECALL_DISPERSION_MIN_ROWS
            || !median.is_finite()
            || !deviation.is_finite()
            || median <= 0.0
            || deviation < median * RECALL_DISPERSION_MIN_RELATIVE_SPREAD
            || deviation > median * RECALL_DISPERSION_MAX_RELATIVE_SPREAD
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

impl Activatable for RecallEntry {
    fn relevance(&self) -> RecallRelevance {
        self.relevance
    }

    fn use_record(&self) -> Option<&KnowledgeUseRecord> {
        self.use_record.as_ref()
    }

    fn situation_coverage(&self, cue: Option<&SituationCue>) -> f64 {
        cue.map_or(NO_SITUATION, |cue| cue.coverage(&self.situation))
    }

    /// Read from the entry's own stored text and provenance, which the scan
    /// already selects. Nothing is stored and nothing extra is read, so a
    /// detector added later applies to every entry ever written.
    fn salience_share(&self) -> f64 {
        SalienceReading::read(&SalienceSource::of(&self.entry)).share()
    }
}

/// One skill offered as a recall candidate (#1154): procedural memory, cued by
/// the prompt rather than searched for.
///
/// **The body never travels.** A skill body is a whole playbook, and the block's
/// economy is that recognition costs less than recall - a line says a procedure
/// exists, and the model reads it only if it decides to. So the candidate
/// carries the name it can be fetched by and the one line that says what it is
/// for, and nothing else of what it holds.
///
/// **An unapproved skill is never a candidate.** Approval (#1155) records that
/// a person agreed the procedure may be followed, and nothing in the system
/// will hand its body over until they have: `builtin_skill_get` refuses one by
/// name. A line offering it is therefore a line the model can only fail on, and
/// it would accrue an offer every turn it ranked near a prompt and never an
/// open - the profile ranking reads as the cleanest evidence to retire an entry.
/// The exclusion is the adapter's, so the spread the arm is read against is the
/// spread of the followable catalog.
#[derive(Debug, Clone)]
pub struct RecallSkill {
    /// The catalog name, which is also the handle the skill is fetched by.
    pub name: String,
    /// The skill's own "when to use" line, as its frontmatter states it.
    pub description: String,
    /// Whether the skill's files were on disk at the last scan of its scope.
    ///
    /// `false` does **not** make the skill unusable, which is why it is marked
    /// on the line rather than excluded from the arm: the catalog is cumulative
    /// (#639), the body still reads, and the procedure is still good. What is
    /// gone is `disk_path` and the attachments, so the skill's bundled scripts
    /// cannot be run.
    pub present_on_disk: bool,
    pub relevance: RecallRelevance,
    /// What the use log knows about this skill (#1154), on the same terms as
    /// [`RecallEntry::use_record`]: the reinforcement half of its activation.
    pub use_record: Option<KnowledgeUseRecord>,
}

impl RecallSkill {
    /// A candidate the use log has nothing to say about.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        present_on_disk: bool,
        relevance: RecallRelevance,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            present_on_disk,
            relevance,
            use_record: None,
        }
    }

    /// The same candidate, carrying what the log knows about it.
    pub fn with_use_record(mut self, record: Option<KnowledgeUseRecord>) -> Self {
        self.use_record = record;
        self
    }
}

impl Activatable for RecallSkill {
    fn relevance(&self) -> RecallRelevance {
        self.relevance
    }

    fn use_record(&self) -> Option<&KnowledgeUseRecord> {
        self.use_record.as_ref()
    }

    /// A skill records no situation yet (#1154). `knowledge_situation` is keyed
    /// on a knowledge entry, and nothing writes a row for a skill, so the term
    /// has nothing to read and contributes exactly zero - which is how every
    /// candidate scored before #1125 existed.
    ///
    /// This is the arm's largest known gap rather than a settled answer. A
    /// procedure is more situational than a fact, not less: "deploy this" is a
    /// weak query and a strong situation, which is the whole reason this arm
    /// exists. Giving a skill a situation record is what would let the cue
    /// reach it.
    fn situation_coverage(&self, _cue: Option<&SituationCue>) -> f64 {
        NO_SITUATION
    }

    /// A skill carries no salience reading (#1127). Every signal is read from a
    /// knowledge entry's own body, summary, tags and provenance, and a skill
    /// holds none of those: its body never travels, and a person approves a
    /// skill rather than promoting it. So the term has nothing to read and
    /// contributes exactly zero.
    ///
    /// Approval (#1155) is the nearest thing the catalog holds to a person's own
    /// instruction, and it is deliberately not read here: every followable skill
    /// is approved, so a signal every candidate carries separates nobody.
    fn salience_share(&self) -> f64 {
        NO_SALIENCE
    }
}

/// What a candidate contributes to its activation score
/// ([`crate::domain::activation`]).
///
/// One ranking rule, read through one trait, so the knowledge arm and the skill
/// arm cannot drift apart. Both hold the same two signals - how far the
/// candidate stands out of its own source, and what the use log knows about it -
/// and the block orders each arm by the same function of them.
pub trait Activatable {
    /// How near this candidate is to the prompt, and in which sense.
    fn relevance(&self) -> RecallRelevance;
    /// What the use log knows about it, where anything does.
    fn use_record(&self) -> Option<&KnowledgeUseRecord>;
    /// How much of what the present situation could have said about this
    /// candidate it did say (#1125), or
    /// [`NO_SITUATION`] where there is
    /// nothing to read.
    ///
    /// A source answers `NO_SITUATION` where it keeps no situation record of
    /// its own, and the term then contributes exactly zero - the same answer a
    /// knowledge entry gets when no cue was measured. That is a statement about
    /// what is recorded, not a judgement that the situation does not matter
    /// here: a procedure is if anything more situational than a fact, which is
    /// why #1154 reads #1125 rather than duplicating it.
    fn situation_coverage(&self, cue: Option<&SituationCue>) -> f64;
    /// How many of the salience signals this build can detect the candidate
    /// carries, as a share (#1127), or
    /// [`NO_SALIENCE`] where the source
    /// holds no text a detector can read - see [`crate::domain::salience`].
    fn salience_share(&self) -> f64;
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
    /// Ceiling on skill catalog rows read.
    pub skill_limit: usize,
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
    /// The skills nearest the prompt, nearest first (#1154).
    pub skills: Vec<RecallSkill>,
    /// The knowledge source's own dispersion, measured over the whole source.
    /// `None` where the adapter could not measure one, which leaves the block on
    /// its stated estimate.
    pub entry_dispersion: Option<RecallDispersion>,
    /// The scratchpad source's own dispersion, on the same terms (#1167).
    ///
    /// Its own, and never the knowledge arm's. A note embeds
    /// `"<key> <content>"`, which is terser and more telegraphic than an
    /// entry's body, so the pad puts its distances somewhere else. `None` is
    /// the ordinary answer rather than the exceptional one, because one
    /// conversation's pad is usually under
    /// [`RECALL_DISPERSION_MIN_ROWS`]; what the measurement buys is the long
    /// conversation, whose pad is both large enough to measure and least like
    /// the store.
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
    /// The skill catalog's own dispersion, on the same terms.
    ///
    /// Its own, and never the knowledge arm's. A skill row embeds a name, a
    /// short "when to use" line and a playbook body, and a knowledge row embeds
    /// a fact - so the two sources put their distances in different places, and
    /// a bar read against one says nothing about the other. That is the rule
    /// [`RecallDispersion`] exists for, and the skill catalog is the case that
    /// makes it visible: it is small, and its rows are shaped unlike anything
    /// else the block reads.
    pub skill_dispersion: Option<RecallDispersion>,
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

    /// Both directions of the measured read, against geometry taken from a real
    /// pad and a real embedding model (#1243).
    ///
    /// **This pins the arithmetic, not the pad arm.** It exercises
    /// [`RecallDispersion`] and [`RecallRelevance::clears_bar`] on fixed values,
    /// so it stays green if the pad arm stops consulting its own measurement
    /// altogether - `a_note_is_read_against_the_pads_own_dispersion` in
    /// `crate::recall` is the test that fails for that. The two cover the
    /// property together and neither covers it alone.
    ///
    /// The estimate is a bar in a fixed place, so it does two wrong things
    /// rather than one: it admits freely when the prompt lands near the pad,
    /// and it admits nothing at all when the pad sits beyond 0.31 however
    /// clearly one note stands out. Naming only the first half would let a
    /// later change turn the pad arm into a one-way filter and still pass.
    #[test]
    fn a_measured_pad_admits_where_the_estimate_is_dark_and_refuses_where_it_floods() {
        let rows = RECALL_DISPERSION_MIN_ROWS;
        let estimate = crate::recall::RECALL_ASSUMED_DISPERSION;
        let bar = crate::recall::RECALL_BAR;

        // A source the bar can read, whose spread is wide enough that a
        // distance inside the estimate's fixed 0.31 is nothing special.
        let spread = RecallDispersion::measured(0.50, 0.07, rows)
            .expect("a seventh of the median is inside the band a measurement may claim");
        let near = RecallRelevance::Distance(0.30);
        assert!(
            near.clears_bar(estimate, bar),
            "0.30 sits inside the estimate's fixed 0.31, so the estimate floods"
        );
        assert!(
            !near.clears_bar(spread, bar),
            "the measurement refuses it: against this source 0.30 is an ordinary row"
        );

        // The geometry a real pad measured on an unrelated prompt: every
        // distance far away and tightly grouped, so the estimate admits nothing
        // at all while the measurement can still name what stands out.
        let tight = RecallDispersion::measured(0.901, 0.028, rows)
            .expect("a thirtieth of the median clears the degenerate-spread floor");
        let far = RecallRelevance::Distance(0.70);
        assert!(
            !far.clears_bar(estimate, bar),
            "0.70 is past the estimate's fixed 0.31, so the estimate is dark"
        );
        assert!(
            far.clears_bar(tight, bar),
            "the measurement admits it: it stands out from a source that is \
             otherwise uniformly further away"
        );
    }

    /// The two spread guards are one rule read from both ends (#1243).
    ///
    /// A spread too narrow admits the whole source; a spread too wide admits
    /// nothing at all, including a distance of zero. Neither can separate one
    /// row from another, so neither is a calibration, and both leave the caller
    /// on its stated estimate.
    #[test]
    fn a_spread_too_wide_to_admit_anything_is_refused_like_one_too_narrow() {
        let rows = RECALL_DISPERSION_MIN_ROWS;
        let bar = crate::recall::RECALL_BAR;

        // The geometry a real scratchpad measured on a prompt about its own
        // subject. Read literally it puts the bar below zero, so a note at
        // distance zero - a perfect match to the prompt - could not clear it.
        let unreadable = RecallDispersion::assumed(0.776, 0.125);
        assert!(
            unreadable.distance_at(bar) < 0.0,
            "the fixture must be a geometry that admits nothing, or this test \
             is not exercising the case it is named for"
        );
        assert!(
            !RecallRelevance::Distance(0.0).clears_bar(unreadable, bar),
            "read literally, not even an exact match clears it"
        );
        assert_eq!(
            RecallDispersion::measured(0.776, 0.125, rows),
            None,
            "so the measurement is refused rather than applied"
        );

        // Either side of the threshold, which is the bar's own reciprocal.
        let median = 0.80;
        let widest = median * RECALL_DISPERSION_MAX_RELATIVE_SPREAD;
        assert!(RecallDispersion::measured(median, widest, rows).is_some());
        assert_eq!(
            RecallDispersion::measured(median, widest * 1.01, rows),
            None
        );
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
