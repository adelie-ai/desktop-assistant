//! One activation score for retrieval (#1123).
//!
//! Retrieval had two scoring stories that never met. The live one ranked on
//! semantic distance and full-text rank, and admitted a candidate on one
//! dimensionless bar ([`crate::recall::RECALL_BAR`]). The measured one -
//! [`crate::domain::knowledge_use`] - turned the use log into a number that no
//! ranking path read. This module is the single function that settles both, so
//! there is one place to read, one place to argue with, and one place to fit.
//!
//! ## The score
//!
//! ```text
//! A_i = semantic + lexical + reinforcement + situation + salience
//! ```
//!
//! - **`semantic`** is how far the candidate stands out of its own source,
//!   counted in that source's median absolute deviations - the quantity
//!   [`RecallDispersion::deviations_below_median`] answers. It is never a raw
//!   distance. A distance depends on the embedding model, on how much text a row
//!   holds, and on how wide the store's subject matter is, so it means nothing
//!   across a source boundary; the deviation count means the same thing against
//!   any source. That is what lets a fourth or a fifth source join later without
//!   refitting every weight.
//! - **`reinforcement`** is what the use log knows, put on that same
//!   dimensionless scale: [`ActivationWeights::use_lift`] deviations for every
//!   e-fold of accumulated use, measured against the sum one use a day old
//!   produces. The ratio is what makes it dimensionless - a base-level sum
//!   scales with whatever unit its ages are counted in, and dividing by a
//!   reference sum cancels that. See [`ActivationWeights::reinforcement`].
//! - **`situation`** is how much of what the present situation could have told
//!   us about this entry it actually told us (#1125), spent against a lift of
//!   exactly one reference use. [`crate::domain::situation`] states the cue, the
//!   fan weighting behind it, and why the bound is a scale;
//!   [`ActivationWeights::situation`] is the two lines that turn it into
//!   deviations.
//! - **`lexical`** is how much of the query's own words the candidate carries,
//!   spent against the spread this query's own source has (#1239). It is the
//!   full-text-rank term the table below listed as awaiting an input for as
//!   long as no caller ran both modes at once; the knowledge-search tool runs
//!   both over one store in one call, so it is the caller that supplies one.
//!   [`ActivationWeights::lexical`] states the equivalence.
//! - **`salience`** is how much of the salience information this build can
//!   detect the entry carries (#1127), spent against the same lift.
//!   [`crate::domain::salience`] states the signals, why they divide one lift
//!   rather than each adding one, and why a reading of text is bounded by one
//!   recorded use; [`ActivationWeights::salience`] turns it into deviations.
//!
//! [`RecallDispersion::deviations_below_median`]:
//!     crate::ports::recall::RecallDispersion::deviations_below_median
//!
//! ## Why base-level learning rather than a hand-tuned blend
//!
//! The reinforcement half is Anderson's ACT-R base-level activation, computed by
//! [`KnowledgeUseRecord::use_sum`]: every use is weighted by its own age raised
//! to a negative power, and the sum is read through a logarithm. Three
//! properties come out of that shape rather than out of a tuning pass.
//!
//! - **It is fitted, not invented.** The power law was derived from human
//!   forgetting curves, which is the closest available answer to "how likely is
//!   this to be wanted now".
//! - **Spacing comes free.** Each use carries its own age, so twenty uses spread
//!   over a year outrank twenty uses in one afternoon a year ago. A lifetime
//!   counter cannot express that, and a plain recency weight loses the frequency
//!   half.
//! - **The logarithm is the cap, rather than a clamp.** Marks raise ranking,
//!   ranking drives retrieval, and retrieval is what lets an entry be marked, so
//!   a linear count compounds. Doubling the accumulated sum adds at most
//!   `use_lift * ln 2` however large that sum already is, which holds the whole
//!   term under [`MAX_REINFORCEMENT_DEVIATIONS`] over any history a store
//!   produces - see [`DEFAULT_USE_LIFT`] for what that buys and what it does
//!   not, because the loose reading of it is false.
//!
//!   **The sum, not the number of events.** One more use raises the sum by more
//!   than a factor of two when it is far more recent than everything before it -
//!   a second open thirty seconds after the first roughly doubles a
//!   one-open history and then some. That is the recency half of the signal
//!   doing its job, and no compression of the frequency half can or should stop
//!   it. What the logarithm rules out is the other loop: an entry that keeps
//!   being retrieved because it was retrieved.
//!
//! ## The terms that have no input yet
//!
//! The full form the epic describes carries three more terms. None of them has
//! anything to read today, so none of them is a parameter here - a weight with
//! no input is a number nobody can fit and everybody has to maintain.
//!
//! | term | where its input will come from |
//! | --- | --- |
//! | interference penalty | the entry disposition of #893, which no column holds yet |
//!
//! Each of them adds to `A_i` when it exists. The semantic term is already
//! dimensionless, so a new term states its own weight in the same deviations and
//! nothing already fitted has to move - which is exactly how the lexical term
//! joined, and what a recall lookup still answers
//! [`NO_LEXICAL`] to, because it uses one mode at a time and so carries no rank
//! (see [`RecallRelevance`]).
//!
//! [`RecallRelevance`]: crate::ports::recall::RecallRelevance

use chrono::{DateTime, Utc};

use crate::domain::knowledge_use::{KnowledgeUseRecord, UseScoreWeights};

/// The age of the one use that anchors the reinforcement scale, in seconds.
///
/// **A base-level sum has no natural size.** Every term is an age raised to
/// `-d`, so the whole sum scales by `unit^d` when the ages are counted in a
/// different unit - stated in seconds it is a hundredth of what the same history
/// states in days. Adding that straight to a deviation count would let an
/// arbitrary choice of unit decide how much the use log is worth, which is the
/// error the semantic half exists to avoid. So the sum is read as a ratio
/// against the sum one use of this age produces, and the unit cancels.
///
/// One day is the anchor because it is the scale a knowledge base is used on:
/// an entry opened yesterday is the ordinary "recently useful" case, and it is
/// worth [`ActivationWeights::use_lift`] times `ln 2` here. Anything opened
/// within the hour scores above it and anything untouched for a month scores
/// below.
pub const USE_REFERENCE_AGE_SECONDS: f64 = 24.0 * 60.0 * 60.0;

/// How many of a source's own median absolute deviations one e-fold of
/// accumulated use is worth, relative to [`USE_REFERENCE_AGE_SECONDS`].
///
/// A deliberate scale rather than a fitted one, and what it is chosen against is
/// the semantic term's own spread. Adjacent candidates over a real store sit a
/// few tenths of a deviation apart, and a candidate the bar admits sits several
/// deviations above one it refuses. At this lift one use a day old is worth
/// about a third of a deviation - enough to settle a near-tie - and an extreme
/// history reaches about three.
///
/// **What the ceiling buys, stated exactly, because the loose version is
/// false.** It buys a bound and nothing more: the term stays under
/// [`MAX_REINFORCEMENT_DEVIATIONS`], so history cannot run away with the
/// ranking and a candidate cannot be carried an unbounded distance up a block.
/// A lead wider than that ceiling cannot be closed by any history at all.
///
/// It does **not** buy the top line unconditionally, and a doc that claims
/// otherwise is wrong on this project's own numbers. The bar is 6.8 by
/// construction, so the weakest candidate in any block sits there; the measured
/// prompts put a real hit between 7.3 and 11.4. The best match therefore leads
/// the bar by anywhere from half a deviation to 4.6, and only the wide end of
/// that range is out of reach of a large history.
///
/// **The narrow end is the design working, not a leak in it.** A best match half
/// a deviation above the bar is a weakly cued prompt - the store held nothing
/// the prompt really named. That is exactly the condition under which what has
/// been used recently should lead, and it is the reason base-level activation is
/// the right shape here rather than a tiebreak bolted onto distance. An entry
/// the assistant has been reading all morning taking the top line on a prompt
/// that brushes it is the behaviour asked for. When the prompt does name
/// something the store holds, the semantic term is several deviations clear and
/// the ceiling keeps that line where it belongs.
pub const DEFAULT_USE_LIFT: f64 = 0.5;

/// The most the reinforcement term reaches over any history a store
/// realistically holds.
///
/// Not a clamp - nothing enforces it, and the accumulated sum has no ceiling of
/// its own. It is a property of the logarithm at [`DEFAULT_USE_LIFT`], stated
/// here so the guarantee that rests on it has one number to read, and pinned by
/// `an_extreme_use_history_stays_inside_the_stated_ceiling` against the largest
/// history worth modelling: hundreds of opens across a year, and a person's mark
/// set a minute ago, which is the single largest term the log can carry.
pub const MAX_REINFORCEMENT_DEVIATIONS: f64 = 3.0;

/// The coefficients `activation` applies.
///
/// A struct rather than constants for the same reason
/// [`UseScoreWeights`] is one: a deployment that has kept a use log can fit its
/// own and pass them in, which is what the log exists for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationWeights {
    /// How many deviations one e-fold of accumulated use is worth. See
    /// [`DEFAULT_USE_LIFT`].
    pub use_lift: f64,
    /// What one use, and one mark, are worth to the base-level sum.
    pub use_score: UseScoreWeights,
}

impl Default for ActivationWeights {
    fn default() -> Self {
        Self {
            use_lift: DEFAULT_USE_LIFT,
            use_score: UseScoreWeights::default(),
        }
    }
}

impl ActivationWeights {
    /// The base-level sum one use at [`USE_REFERENCE_AGE_SECONDS`] produces.
    ///
    /// The denominator [`Self::reinforcement`] divides by, so that what the use
    /// log is worth is stated against a use anybody can picture rather than
    /// against the unit the ages happen to be counted in. It follows the decay
    /// exponent, because the sum does.
    pub fn reference_sum(&self) -> f64 {
        USE_REFERENCE_AGE_SECONDS.powf(-self.use_score.safe_decay())
    }

    /// What an accumulated use sum is worth, in the source's own deviations.
    ///
    /// `use_lift * ln(1 + |sum| / reference)`, carrying the sum's sign. Four
    /// things the shape has to answer, and it answers each by construction
    /// rather than by a clamp:
    ///
    /// - **An entry nothing has used contributes nothing.** The sum is zero and
    ///   so is the answer, so a cold store ranks exactly as it did before this
    ///   score existed. No floor constant decides how far below a used entry an
    ///   unused one sits.
    /// - **One use at the reference age is worth `use_lift * ln 2`.** That is
    ///   the anchor, and it is what makes the answer independent of the unit the
    ///   ages are counted in.
    /// - **A negative mark subtracts.** "Offered, opened, and it was wrong"
    ///   drives the sum below zero, so a refuted entry ends below one nobody has
    ///   ever opened, rather than merely level with it. It does not subtract as
    ///   much as the same mark would add, and that is the writer's doing rather
    ///   than this function's: a mark is recorded as a use whichever way it
    ///   points, because the entry really was retrieved, so a negative mark of
    ///   weight `w` nets `w - 1` against a positive one's `w + 1`. The function
    ///   is symmetric about zero; the input it is given is not.
    /// - **Doubling the sum adds at most `use_lift * ln 2`.** That is the cap
    ///   #698 asks for, as a property of the function rather than a clamp. It
    ///   is a statement about the accumulated sum and not about the number of
    ///   uses: one more use raises the sum by over a factor of two when it is
    ///   far more recent than everything before it, and that is recency, which
    ///   is a signal this score is meant to carry.
    pub fn reinforcement(&self, sum: f64) -> f64 {
        let reference = self.reference_sum();
        if !sum.is_finite() || !reference.is_finite() || reference <= 0.0 {
            return 0.0;
        }
        self.use_lift * (sum.abs() / reference).ln_1p() * sum.signum()
    }

    /// What one use at [`USE_REFERENCE_AGE_SECONDS`] is worth, in the source's
    /// own deviations.
    ///
    /// **The scale every cheap signal is spent against**, and the whole answer
    /// to "why that number" for each of them. Computed from
    /// [`Self::reinforcement`] rather than restated, so no two of them can drift
    /// and there is one definition to argue with.
    ///
    /// It is a scale rather than a fit, and it introduces no coefficient of its
    /// own. What it states is an equivalence between two signals - a cheap one
    /// at full strength is worth what "you opened this yesterday" is worth - and
    /// an equivalence transfers to a store nobody measured, which is precisely
    /// what [`DEFAULT_USE_LIFT`]'s value does not do. Three properties follow:
    ///
    /// - **No unit of its own.** It works out to `use_lift * ln 2` whatever the
    ///   decay exponent and whatever unit the use log's ages are counted in,
    ///   because the reinforcement term is already a ratio against its own
    ///   reference. It follows [`USE_REFERENCE_AGE_SECONDS`], which is a genuine
    ///   unit normalization, rather than [`DEFAULT_USE_LIFT`], which is not.
    /// - **One number to fit, not several.** A deployment that fits `use_lift`
    ///   from its own use log moves every term that reads this together, and
    ///   keeps the stated relations.
    /// - **A size that suits what it claims.** At the default weights it is
    ///   about a third of a deviation, which settles a near-tie between adjacent
    ///   candidates and is a ninth of [`MAX_REINFORCEMENT_DEVIATIONS`].
    ///
    ///   **What one such signal can and cannot do, exactly.** A third of a
    ///   deviation is under the half a deviation that separates the bar from the
    ///   weakest hit the measured prompts produced, so one cheap signal on its
    ///   own cannot take the top line from any measured best match. Two of them
    ///   together reach about seven tenths and can. That is not a leak: a best
    ///   match half a deviation above the bar means the prompt named nothing the
    ///   store really holds, and a weakly cued prompt is exactly when the
    ///   non-semantic signals should lead - the same case
    ///   [`DEFAULT_USE_LIFT`] describes for the use log, and the guarantee
    ///   #1123 deliberately did not make. Beyond about seven tenths of a
    ///   deviation the semantic lead stands against both cheap signals together.
    ///   The use log is not one of them and is not bounded by this: see
    ///   [`MAX_REINFORCEMENT_DEVIATIONS`].
    pub fn reference_use_lift(&self) -> f64 {
        self.reinforcement(self.reference_sum())
    }

    /// The most a full situation match is worth, in the source's own
    /// deviations: **exactly what one use at [`USE_REFERENCE_AGE_SECONDS`] is
    /// worth**.
    ///
    /// [`Self::reference_use_lift`] holds the whole argument. The equivalence
    /// this name states is that "this entry recurs where you are now" is worth
    /// what "you opened this yesterday" is worth.
    pub fn situation_lift(&self) -> f64 {
        self.reference_use_lift()
    }

    /// The most a full salience reading is worth, in the source's own
    /// deviations: **exactly what one use at [`USE_REFERENCE_AGE_SECONDS`] is
    /// worth**, the same lift the situation gets.
    ///
    /// [`Self::reference_use_lift`] holds the whole argument. The equivalence
    /// this name states is that "this entry carries everything that makes a fact
    /// worth keeping" is worth what "you opened this yesterday" is worth.
    ///
    /// **Why the reading is not worth more than one recorded use**, when one of
    /// the signals may be a person's own doing: a mark in the use log records a
    /// judgement somebody made, and a salience signal is inferred from what an
    /// entry happens to look like. A reading must not outweigh a record. See
    /// [`crate::domain::salience`].
    pub fn salience_lift(&self) -> f64 {
        self.reference_use_lift()
    }

    /// What a situation coverage is worth, in the source's own deviations.
    ///
    /// `situation_lift * coverage`, over a coverage in `[0, 1]` -
    /// [`SituationCue::coverage`] answers that range and this holds it to the
    /// range whatever it is handed, so the bound is a property of the function
    /// and not of its caller.
    ///
    /// Never negative. An entry the cue cannot grade, and an entry whose record
    /// disagrees with the cue outright, both contribute exactly zero, so neither
    /// can end below what its own distance and history earned it. Mismatch
    /// forfeits the lift; it does not subtract.
    ///
    /// [`SituationCue::coverage`]: crate::domain::situation::SituationCue::coverage
    pub fn situation(&self, coverage: f64) -> f64 {
        if !coverage.is_finite() || coverage <= 0.0 {
            return 0.0;
        }
        // `max(0.0)` rather than a bare product, so "never negative" is a
        // property of this function and not of the weights it happens to be
        // handed. Nothing constructs these weights but `Default` today; the
        // struct exists so a deployment can fit its own from its own use log,
        // and a negative `use_lift` would otherwise make a matching situation
        // subtract - the opposite of what every line above claims.
        self.situation_lift().max(0.0) * coverage.min(1.0)
    }

    /// What one query's own words are worth, in the source's own deviations.
    ///
    /// `spread * share`, over a share in `[0, 1]` - and this holds it to that
    /// range whatever it is handed, so the bound is a property of the function
    /// and not of its caller.
    ///
    /// **The equivalence, stated so it can be argued with.** A row that carries
    /// the query's words better than anything else in the store stands as far
    /// out of that store as its nearest row stands from its furthest. It is a
    /// scale rather than a fit for the reason [`Self::reference_use_lift`] is:
    /// it states a relation between two signals and computes the number, and it
    /// introduces no coefficient of its own. Both factors are measured over the
    /// same source in the same pass, so nothing here is carried from one
    /// deployment to another.
    ///
    /// **Why the spread rather than one reference use.** The two cheap signals
    /// are bounded by what one day-old use is worth, about a third of a
    /// deviation, because each is a reading of something the entry happens to
    /// look like. This is not that. A row the query names exactly, and that
    /// nothing else in the store names, is the strongest evidence a text search
    /// has - and a third of a deviation cannot move it off the bottom of a
    /// page: measured on a seeded store, such a row sat thirteenth by distance,
    /// and a lift that size left it thirteenth. A term too small to change an
    /// order is a term that is not there.
    ///
    /// **Why the spread rather than the nearest row's own standing.** That
    /// would make a full lexical match tie the best semantic match rather than
    /// lead it, and a tie is settled by whatever the sort visited first - so
    /// the answer would depend on the scan's order instead of on the score.
    ///
    /// Never negative. A row the query's words did not reach contributes
    /// exactly zero, so a search over a store with no full-text hit ranks
    /// exactly as it ranked before this term existed. Absence forfeits the
    /// lift; it does not subtract.
    ///
    /// The spread is an extreme rather than a robust statistic, and that is a
    /// deliberate exception to the rule the median and the deviation follow. It
    /// is a **ceiling on a bounded share** here, not a unit anything is divided
    /// by, so an unusual row at either end widens a lift rather than corrupting
    /// every score - and the share is zero for every row the query's words did
    /// not reach, so a widened ceiling reaches nobody who did not match.
    pub fn lexical(&self, lexical: LexicalMatch) -> f64 {
        if !lexical.share.is_finite() || lexical.share <= 0.0 || !lexical.spread.is_finite() {
            return 0.0;
        }
        lexical.spread.max(0.0) * lexical.share.min(1.0)
    }

    /// What a salience reading is worth, in the source's own deviations.
    ///
    /// `salience_lift * share`, over a share in `[0, 1]` -
    /// [`SalienceReading::share`] answers that range and this holds it to the
    /// range whatever it is handed, so the bound is a property of the function
    /// and not of its caller.
    ///
    /// Never negative. An entry that carries no signal contributes exactly
    /// zero, so a store this detector says nothing about ranks exactly as it
    /// ranked before the term existed. Absence forfeits the lift; it does not
    /// subtract, because "no deadline, no money, nobody asked for it" describes
    /// most of what is worth keeping.
    ///
    /// [`SalienceReading::share`]: crate::domain::salience::SalienceReading::share
    pub fn salience(&self, share: f64) -> f64 {
        if !share.is_finite() || share <= 0.0 {
            return 0.0;
        }
        // `max(0.0)` rather than a bare product, for the reason
        // [`Self::situation`] gives: a deployment may fit its own `use_lift`,
        // and a negative one would otherwise make a salient entry subtract.
        self.salience_lift().max(0.0) * share.min(1.0)
    }
}

/// What one query's own words found, and how far this source lets anything
/// stand out for that query (#1239).
///
/// **The full-text signal has no dispersion of its own that can be measured.**
/// A `ts_rank` is a bare number, like a raw cosine distance, and it cannot
/// cross a source boundary - but the obvious mirror of the semantic term does
/// not work either: a full-text query usually matches a handful of rows, and
/// the median absolute deviation of a handful is noise. Worse, the case this
/// term exists for is the one where **one** row matches, where a spread over
/// the matched set is not merely noisy but undefined.
///
/// So the two halves are measured separately, and both against the source:
///
/// - [`Self::share`] is where the row stands among the rows this query's words
///   did reach - a ratio, so it carries no unit and needs no spread of its own.
/// - [`Self::spread`] is how many of the source's own median absolute
///   deviations separate its nearest row from its furthest, for this query. It
///   is what turns the ratio into deviations, and it is a property of the
///   source and of the query together, so it is measured in the pass that ranks
///   and never cached.
///
/// [`ActivationWeights::lexical`] states what the two are worth together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalMatch {
    /// Where this candidate stands among the rows the query's own words
    /// reached, in `[0, 1]`: one for the best of them, zero for a row those
    /// words did not reach at all.
    ///
    /// A ratio against this query's own best match rather than a raw rank, for
    /// the reason the type's own documentation gives. A row the full-text arm
    /// never returned carries [`NO_LEXICAL`].
    pub share: f64,
    /// How many of the source's own median absolute deviations separate its
    /// nearest row from its furthest, for this query.
    ///
    /// Measured over every row the scan could reach, never over the rows it
    /// returned: the returned rows are the near tail, which is the part a cued
    /// query moves.
    pub spread: f64,
}

impl LexicalMatch {
    /// A candidate the query's own words did not reach, which is what every
    /// caller with no full-text arm passes.
    ///
    /// Named rather than written as a pair of zeroes at each call site, for the
    /// reason [`NO_SITUATION`] is. A recall lookup uses one mode at a time, so
    /// its candidates carry either a distance or a rank and never both - see
    /// [`RecallRelevance`](crate::ports::recall::RecallRelevance).
    pub const NONE: Self = Self {
        share: NO_LEXICAL,
        spread: 0.0,
    };
}

/// A share for a candidate the query's own words did not reach.
///
/// Named rather than written as a bare zero, for the reason [`NO_SITUATION`]
/// is.
pub const NO_LEXICAL: f64 = 0.0;

/// A candidate the situation cannot grade, which is what every caller passes
/// where no cue was measured or the entry has no record of its own.
///
/// Named rather than written as a bare zero at each call site, so a reader meets
/// "no situation signal" instead of a number whose meaning depends on the
/// parameter it sits in.
pub const NO_SITUATION: f64 = 0.0;

/// A candidate no salience detector says anything about, which is what every
/// caller passes where a source keeps no readable text of its own.
///
/// Named rather than written as a bare zero at each call site, for the reason
/// [`NO_SITUATION`] is.
pub const NO_SALIENCE: f64 = 0.0;

/// What one candidate is worth, as of `now`.
///
/// `semantic` is the source-normalized signal: how many of its own source's
/// median absolute deviations the candidate stands below that source's median.
/// `record` is what the use log knows about it, and `None` - an entry nothing
/// has ever offered, opened or marked - contributes exactly zero, so a store
/// with no use history ranks on the semantic signal alone.
/// `situation_coverage` is how much of what the present situation could have
/// told us about this entry it did tell us
/// ([`SituationCue::coverage`](crate::domain::situation::SituationCue::coverage)),
/// and [`NO_SITUATION`] is the answer wherever there is no cue to read.
/// `salience_share` is how much of the salience information this build can
/// detect the entry carries
/// ([`SalienceReading::share`](crate::domain::salience::SalienceReading::share)),
/// and [`NO_SALIENCE`] is the answer for a source that carries no readable text.
/// `lexical` is how much of the query's own words the candidate carries and how
/// far this source lets anything stand out for that query (#1239), and
/// [`LexicalMatch::NONE`] is the answer wherever there is no full-text arm to
/// read - which is every recall lookup, because one uses one mode at a time.
///
/// Every extra signal is handed in already dimensionless, exactly as the
/// semantic one is. That is what lets a fourth source, or a fifth term, join
/// later without refitting anything already here.
/// **Crate-private on purpose** (#1244). Every path that ranks by activation
/// supplies its terms through
/// [`Activatable`](crate::ports::recall::Activatable), and that is only a
/// mechanism while this function cannot be reached around it. Exported, a
/// ranking site in an adapter crate could call it directly and answer a term
/// with a literal - which is precisely how the search path came to pass
/// `NO_SITUATION` and rank on three terms while the block ranked on four.
///
/// Widening this back to `pub` reopens that door, so if a caller outside this
/// crate needs to rank, give it an `Activatable` implementation rather than
/// this function.
///
/// **Test-only from #1327 on.** [`rank_by_activation_traced`](crate::ports::recall::rank_by_activation_traced)
/// reads [`activation_terms`] directly, so this scalar form has no production
/// caller left; it stays as the fixture the arithmetic tests below and in
/// [`crate::domain::replay`] pin their expectations against, and
/// [`activation_terms`] is what it now delegates to.
#[cfg(test)]
pub(crate) fn activation(
    semantic: f64,
    record: Option<&KnowledgeUseRecord>,
    situation_coverage: f64,
    salience_share: f64,
    lexical: LexicalMatch,
    now: DateTime<Utc>,
    weights: &ActivationWeights,
) -> f64 {
    activation_terms(
        Some(semantic),
        record,
        situation_coverage,
        salience_share,
        lexical,
        now,
        weights,
    )
    .total
}

/// The [`ActivationWeights`] shape a stored [`ActivationTerms`] was scored
/// under, bumped whenever a term is added or a term's meaning changes.
///
/// A stored row's `weights` travel with this string (#1327) so a reader
/// comparing an old turn's terms against today's ranking knows whether the two
/// are the same computation or a different one wearing the same field names.
pub const ACTIVATION_SCORER_VERSION: &str = "1327-v1";

/// One candidate's activation score, kept broken out by term (#1327).
///
/// [`crate::ports::recall::rank_by_activation_traced`] is the only place this
/// is built, and [`activation`] reads its [`Self::total`] rather than
/// recomputing one - so the number a reader is shown and the number the
/// ranking used are the same value, not two computations that are supposed to
/// agree.
///
/// Every field but `semantic` and `total` is a plain deviation, never
/// negative except where the term's own weight function says a negative mark
/// may carry through (see [`ActivationWeights::reinforcement`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationTerms {
    /// How many of its own source's median absolute deviations the candidate
    /// stands below that source's median, or `None` for a candidate with no
    /// semantic signal - a full-text match, which carries a match instead of a
    /// distance (see [`RecallRelevance::semantic_signal`] in
    /// `crate::ports::recall`).
    pub semantic: Option<f64>,
    /// How much of the query's own words the candidate carries, and how far
    /// its source lets anything stand out for that query (#1239). Zero for
    /// every candidate on the recall path today, which carries no full-text
    /// arm - recorded rather than assumed, so a reader sees that the term ran
    /// and answered zero, not that it never ran.
    pub lexical: f64,
    /// What the use log's base-level sum is worth, in deviations (#1123).
    pub reinforcement: f64,
    /// How much of what the present situation could have told us about this
    /// candidate it did tell us (#1125), in deviations.
    pub situation: f64,
    /// How much of the salience information this build can detect the
    /// candidate carries (#1127), in deviations.
    pub salience: f64,
    /// `semantic.unwrap_or(0.0) + lexical + reinforcement + situation +
    /// salience` - what [`rank_by_activation`](crate::ports::recall::rank_by_activation)
    /// sorts by.
    pub total: f64,
}

/// Compute every term of one candidate's activation score, so the score and
/// its record are one computation (#1327).
///
/// `semantic` is `None` for a candidate with no semantic signal to add - see
/// [`ActivationTerms::semantic`]. The other four terms are computed
/// regardless, because a candidate that never carried a semantic reading can
/// still have used the store, been marked, or matched the situation, and the
/// plan this feeds records what each term actually answered rather than
/// leaving it blank for want of one input.
///
/// [`activation`] is this function's `total`, over a `semantic` that is
/// always `Some` - every caller of `activation` already has a distance to
/// read, so its signature keeps that value required rather than optional.
pub(crate) fn activation_terms(
    semantic: Option<f64>,
    record: Option<&KnowledgeUseRecord>,
    situation_coverage: f64,
    salience_share: f64,
    lexical: LexicalMatch,
    now: DateTime<Utc>,
    weights: &ActivationWeights,
) -> ActivationTerms {
    let sum = record.map_or(0.0, |record| record.use_sum(now, &weights.use_score));
    let lexical_term = weights.lexical(lexical);
    let reinforcement_term = weights.reinforcement(sum);
    let situation_term = weights.situation(situation_coverage);
    let salience_term = weights.salience(salience_share);
    let total = semantic.unwrap_or(0.0)
        + lexical_term
        + reinforcement_term
        + situation_term
        + salience_term;
    ActivationTerms {
        semantic,
        lexical: lexical_term,
        reinforcement: reinforcement_term,
        situation: situation_term,
        salience: salience_term,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::knowledge_use::{
        KnowledgeMark, MarkPolarity, MarkSource, RECENT_USE_WINDOW,
    };
    use chrono::TimeDelta;

    const DAY: i64 = 24 * 3600;
    const YEAR: i64 = 365 * DAY;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-07T12:00:00Z")
            .expect("a fixed clock parses")
            .with_timezone(&Utc)
    }

    fn at(now: DateTime<Utc>, seconds_ago: i64) -> DateTime<Utc> {
        now - TimeDelta::seconds(seconds_ago)
    }

    /// A record of `opens` opens, the newest [`RECENT_USE_WINDOW`] of `ages`
    /// held exactly, first seen when the oldest of them landed.
    ///
    /// The window is sorted youngest-first and cut to the newest, whatever order
    /// `ages` arrives in, because that is what the writer stores: it prepends
    /// `NOW()` and cuts the tail.
    fn used(now: DateTime<Utc>, ages: &[i64], opens: u64) -> KnowledgeUseRecord {
        let mut newest_first = ages.to_vec();
        newest_first.sort_unstable();
        newest_first.truncate(RECENT_USE_WINDOW);
        KnowledgeUseRecord {
            entry_id: "kb-1".to_string(),
            offered_count: opens,
            opened_count: opens,
            marked_count: 0,
            first_seen_at: at(now, ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(at(now, 1)),
            recent_uses: newest_first.iter().map(|a| at(now, *a)).collect(),
            marks: Vec::new(),
        }
    }

    /// `count` uses laid down evenly over the last `span` seconds.
    fn evenly_over(now: DateTime<Utc>, count: u64, span: i64) -> KnowledgeUseRecord {
        let step = span / count as i64;
        let ages: Vec<i64> = (1..=count as i64).map(|i| i * step).collect();
        used(now, &ages, count)
    }

    /// `count` uses at a fixed one-hour spacing, the newest an hour ago: the
    /// same rhythm whatever the count, so a longer history is more use rather
    /// than denser use.
    fn every_hour(now: DateTime<Utc>, count: u64) -> KnowledgeUseRecord {
        let ages: Vec<i64> = (1..=count as i64).map(|i| i * 3_600).collect();
        used(now, &ages, count)
    }

    /// Acceptance (#1123): the base-level term is arithmetic over stored
    /// counters and stored timestamps. No model, no network, no clock but the
    /// one handed in.
    #[test]
    fn base_level_activation_is_computed_from_stored_counters_with_no_model_call() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[60, 600, 6_000], 3);

        let first = activation(
            7.0,
            Some(&record),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let again = activation(
            7.0,
            Some(&record),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert_eq!(
            first, again,
            "the score is a pure function of what it is given"
        );
        assert!(first.is_finite());
        assert!(
            first > 7.0,
            "three recent opens must raise the candidate above its semantic signal alone, \
             got {first}"
        );
    }

    /// Acceptance (#1123): spacing. Both entries have been known for a year and
    /// both were used twenty times; one was used steadily across the year, the
    /// other twenty times inside a single day at the start of it.
    ///
    /// The lifetimes are held equal on purpose. Comparing a spread history
    /// against a burst that happened *today* would measure recency, which every
    /// decay rule has - the spacing effect is what a per-use age buys over a
    /// lifetime counter, and it only shows against an equal elapsed time.
    #[test]
    fn twenty_uses_spread_over_a_year_rank_above_twenty_uses_in_one_day() {
        let now = now();
        let weights = ActivationWeights::default();

        let spread = evenly_over(now, 20, YEAR);
        // Twenty uses inside one day, a year ago: the same count, the same age
        // since the first of them, all of it massed.
        let massed = {
            let ages: Vec<i64> = (0..20).map(|i| YEAR - i * (DAY / 20)).collect();
            used(now, &ages, 20)
        };

        assert_eq!(spread.total_uses(), massed.total_uses(), "precondition");
        assert_eq!(
            spread.first_seen_at, massed.first_seen_at,
            "precondition: equal elapsed time, so only the spacing differs"
        );

        let spread_score = activation(
            7.0,
            Some(&spread),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let massed_score = activation(
            7.0,
            Some(&massed),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        assert!(
            spread_score > massed_score,
            "twenty uses spread over a year scored {spread_score}, twenty massed into a day \
             scored {massed_score}"
        );
    }

    /// Acceptance (#1123): doubling an entry's accumulated use buys a bounded
    /// amount, which is what stops the retrieve-mark-retrieve loop compounding.
    ///
    /// Both halves of the claim. First the function: doubling the accumulated
    /// sum adds at most `use_lift * ln 2`, at every size of sum. Then a record:
    /// an entry used twice as often over twice as long - the same rhythm, kept
    /// up for longer, which is how an entry accrues use over time - gains no
    /// more than that.
    ///
    /// See `one_much_more_recent_use_may_raise_the_score_by_more_than_that` for
    /// the case this bound deliberately does not cover.
    #[test]
    fn doubling_the_accumulated_use_raises_activation_by_a_bounded_amount() {
        let now = now();
        let weights = ActivationWeights::default();
        let bound = weights.use_lift * std::f64::consts::LN_2;
        let reference = weights.reference_sum();

        for exponent in -6..=12 {
            let sum = reference * 2.0_f64.powi(exponent);
            let step = weights.reinforcement(sum * 2.0) - weights.reinforcement(sum);
            assert!(
                step <= bound,
                "doubling a sum of {sum} added {step}, past the {bound} a doubling may buy"
            );
        }

        let mut previous = activation(
            0.0,
            Some(&every_hour(now, 1)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        for count in [2u64, 4, 8, 16, 32, 64, 128, 256] {
            let doubled = activation(
                0.0,
                Some(&every_hour(now, count)),
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights,
            );
            let step = doubled - previous;
            assert!(
                step <= bound,
                "going to {count} uses added {step}, past the {bound} a doubling may buy"
            );
            previous = doubled;
        }
    }

    /// The bound above is about the accumulated sum, and this is the case it
    /// does not cover: adding one use far more recent than everything before it
    /// more than doubles the sum, so the score may rise by more than
    /// `use_lift * ln 2`.
    ///
    /// Stated as a test rather than left as a gap, because a later reader
    /// meeting it in production would otherwise take it for the bound leaking.
    /// It is recency, which is half of what a per-use age buys, and no
    /// compression of the frequency half should suppress it. What bounds this
    /// case is the ceiling, not the step - see
    /// `an_extreme_use_history_stays_inside_the_stated_ceiling`.
    #[test]
    fn one_much_more_recent_use_may_raise_the_score_by_more_than_that() {
        let now = now();
        let weights = ActivationWeights::default();
        let bound = weights.use_lift * std::f64::consts::LN_2;

        let once = activation(
            0.0,
            Some(&used(now, &[60], 1)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let twice = activation(
            0.0,
            Some(&used(now, &[30, 60], 2)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            twice - once > bound,
            "a second open thirty seconds after the first added {}, and recency is supposed to \
             be able to add more than the {bound} a doubled sum buys",
            twice - once
        );
    }

    /// The bar every candidate has already cleared, and so the score the
    /// weakest one in any block carries. `crate::recall::RECALL_BAR` states it;
    /// it is repeated here because this module must not depend on the block.
    const BAR: f64 = 6.8;

    /// What the measured prompts reached, nearest candidate first
    /// (`crate::recall`'s seeded corpus). The weakest real hit reached 7.3 and
    /// the strongest 11.4, so a best match leads the bar by anywhere from half a
    /// deviation to 4.6 - which is the range any claim about the top line has to
    /// hold over.
    const MEASURED_HITS: &[f64] = &[
        11.4, 10.9, 10.2, 9.8, 9.4, 9.1, 8.7, 8.4, 8.0, 7.8, 7.5, 7.4, 7.3,
    ];

    /// An entry opened and marked for years: the largest history a store
    /// realistically holds.
    fn a_veteran(now: DateTime<Utc>) -> KnowledgeUseRecord {
        let mut veteran = evenly_over(now, 500, YEAR);
        veteran.marked_count = 20;
        veteran.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Positive,
            reason: None,
            marked_at: at(now, 60),
        }];
        veteran
    }

    /// Acceptance (#1123), and #698's caution about the top line: a match that
    /// leads the bar by more than [`MAX_REINFORCEMENT_DEVIATIONS`] keeps its
    /// line against any history at all.
    ///
    /// Over every distance the measured prompts reached that leads by that much,
    /// not only the strongest of them: a claim tested at one favourable point is
    /// not a claim. The boundary itself is asserted too, because it is the
    /// number the guarantee is stated in.
    #[test]
    fn a_lead_wider_than_the_ceiling_cannot_be_closed_by_any_use_history() {
        let now = now();
        let weights = ActivationWeights::default();
        let veteran = a_veteran(now);
        let at_the_bar = activation(
            BAR,
            Some(&veteran),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        let wide: Vec<f64> = MEASURED_HITS
            .iter()
            .copied()
            .filter(|hit| hit - BAR > MAX_REINFORCEMENT_DEVIATIONS)
            .collect();
        assert!(
            wide.len() >= 3,
            "the measured corpus must hold several clearly-cued prompts, or this proves \
             nothing: {wide:?}"
        );

        for best in wide
            .iter()
            .chain(std::iter::once(&(BAR + MAX_REINFORCEMENT_DEVIATIONS)))
        {
            let cold = activation(
                *best,
                None,
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights,
            );
            assert!(
                at_the_bar < cold,
                "a veteran at the bar scored {at_the_bar} against a cold best match at \
                 {best} deviations, which scored {cold}"
            );
        }
    }

    /// The other end of the same range, pinned as intended behaviour rather than
    /// left as an exception to a guarantee that does not hold there.
    ///
    /// The weakest hit the measured prompts produced stands 7.3 deviations out,
    /// half a deviation above the bar. A prompt whose best match is that close to
    /// the floor named nothing the store really holds - and a weakly cued prompt
    /// is exactly when what has been used recently should lead. An entry the
    /// assistant has been reading all morning takes the top line here, and that
    /// is the whole reason for choosing base-level activation over a tiebreak
    /// bolted onto distance.
    ///
    /// Stated as a test so that a later change which clamps the term, or scales
    /// it down until it cannot do this, fails here and has to argue with the
    /// design rather than quietly undo it.
    #[test]
    fn a_weakly_cued_prompt_lets_use_history_lead() {
        let now = now();
        let weights = ActivationWeights::default();

        // Ten opens inside the last half hour: an entry in the current working
        // context, which is the ordinary case rather than an extreme one.
        let worked_all_morning = evenly_over(now, 10, 1_800);

        let best_cold_match = activation(
            7.3,
            None,
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let just_above_the_bar = activation(
            6.9,
            Some(&worked_all_morning),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            just_above_the_bar > best_cold_match,
            "the weakest measured hit scored {best_cold_match} and an entry the work keeps \
             needing, four tenths of a deviation below it, scored {just_above_the_bar}; on a \
             prompt this weakly cued the used entry is supposed to lead"
        );
    }

    /// Acceptance (#1123): the semantic term is the source-normalized deviation
    /// and never a raw distance. Two sources of quite different geometry, one
    /// candidate equally placed in each - the score is the same number.
    #[test]
    fn the_semantic_term_is_the_source_normalized_deviation_and_not_a_raw_distance() {
        use crate::ports::recall::RecallDispersion;

        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[600], 1);

        let tight = RecallDispersion::assumed(0.80, 0.05);
        let loose = RecallDispersion::assumed(0.42, 0.30);
        let stands_out_by = 7.5;

        // The same position in each source's own geometry, which is a different
        // raw distance in each.
        let near = tight.distance_at(stands_out_by);
        let far = loose.distance_at(stands_out_by);
        assert!(
            (near - far).abs() > 0.5,
            "precondition: the two sources put an equally exceptional row at very different \
             distances ({near} against {far})"
        );

        let in_tight = activation(
            tight.deviations_below_median(near),
            Some(&record),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let in_loose = activation(
            loose.deviations_below_median(far),
            Some(&record),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            (in_tight - in_loose).abs() < 1e-9,
            "an equally-placed candidate scored {in_tight} against one source and {in_loose} \
             against another"
        );
    }

    /// Acceptance (#1123): an entry the log has never seen is ranked on the
    /// semantic signal alone, so a store with no history ranks exactly as it did
    /// before this score existed.
    #[test]
    fn an_entry_with_no_recorded_use_is_ranked_on_the_semantic_signal_alone() {
        let now = now();
        let weights = ActivationWeights::default();

        for semantic in [0.0, 6.8, 11.4, -3.0] {
            assert_eq!(
                activation(
                    semantic,
                    None,
                    NO_SITUATION,
                    NO_SALIENCE,
                    LexicalMatch::NONE,
                    now,
                    &weights
                ),
                semantic,
                "an unused entry must contribute nothing of its own"
            );
        }

        // A row of zeroes is the same state as no row at all, and must score
        // the same - the two differ only in whether a write ever happened.
        let unseen = KnowledgeUseRecord::unseen("kb-1", now);
        assert_eq!(
            activation(
                6.8,
                Some(&unseen),
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights
            ),
            activation(
                6.8,
                None,
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights
            )
        );
    }

    /// A negative mark drives the entry below one nobody has ever opened, rather
    /// than merely failing to lift it.
    #[test]
    fn a_refuted_entry_ranks_below_one_nobody_has_opened() {
        let now = now();
        let weights = ActivationWeights::default();

        // As the writer stores it: a mark is a use whichever way it points, so
        // the stamp is in the window and `marked_count` moved too.
        let mut refuted = used(now, &[60, 3_600], 1);
        refuted.marked_count = 1;
        refuted.marks = vec![KnowledgeMark {
            source: MarkSource::Model,
            polarity: MarkPolarity::Negative,
            reason: Some("the fact it states was withdrawn".to_string()),
            marked_at: at(now, 60),
        }];

        assert!(
            activation(
                7.0,
                Some(&refuted),
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights,
            ) < activation(
                7.0,
                None,
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights
            )
        );
    }

    /// The reinforcement term's own edges, stated where the reader meets them.
    #[test]
    fn the_reinforcement_term_is_zero_at_zero_and_signed_beyond_it() {
        let weights = ActivationWeights::default();
        let reference = weights.reference_sum();

        assert_eq!(weights.reinforcement(0.0), 0.0);
        assert!(weights.reinforcement(reference) > 0.0);
        assert!(weights.reinforcement(-reference) < 0.0);
        assert_eq!(
            weights.reinforcement(reference),
            -weights.reinforcement(-reference)
        );
        // A sum that is not a number contributes nothing rather than poisoning
        // every comparison the ranking makes.
        assert_eq!(weights.reinforcement(f64::NAN), 0.0);
        assert_eq!(weights.reinforcement(f64::INFINITY), 0.0);
    }

    /// The anchor, pinned: one use at the reference age is worth exactly
    /// `use_lift * ln 2`.
    ///
    /// This is what makes the term dimensionless. A base-level sum scales with
    /// whatever unit its ages are counted in - the same history is a hundred
    /// times larger stated in days than in seconds - so adding one straight to a
    /// deviation count would let that choice decide how much the use log is
    /// worth. Dividing by the sum of one reference use cancels it.
    #[test]
    fn one_use_at_the_reference_age_is_worth_a_stated_amount() {
        let now = now();
        let weights = ActivationWeights::default();
        let a_day = USE_REFERENCE_AGE_SECONDS as i64;

        let scored = activation(
            0.0,
            Some(&used(now, &[a_day], 1)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            (scored - weights.use_lift * std::f64::consts::LN_2).abs() < 1e-9,
            "one use a day old scored {scored}"
        );
    }

    /// The lift has to be large enough to do something. A day-old use must
    /// settle a near-tie - two candidates a few hundredths of a deviation apart,
    /// which is what adjacent rows of a real store look like.
    #[test]
    fn a_recent_use_settles_a_near_tie_between_two_candidates() {
        let now = now();
        let weights = ActivationWeights::default();
        let a_day = USE_REFERENCE_AGE_SECONDS as i64;

        let nearer_but_unread = activation(
            9.10,
            None,
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let further_but_used = activation(
            9.00,
            Some(&used(now, &[a_day], 1)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            further_but_used > nearer_but_unread,
            "a day-old use scored {further_but_used} against an unread candidate a tenth of a \
             deviation nearer at {nearer_but_unread}; a lift this small settles nothing"
        );
    }

    /// [`MAX_REINFORCEMENT_DEVIATIONS`], pinned so the guarantee resting on it
    /// cannot rot.
    ///
    /// The largest history worth modelling - hundreds of opens across a year,
    /// and a person's mark set a minute ago, which is the single largest term
    /// the log can carry - must stay under it. Nothing enforces the bound; it is
    /// a property of the logarithm, and this is what checks the property is
    /// still true.
    #[test]
    fn an_extreme_use_history_stays_inside_the_stated_ceiling() {
        let now = now();
        let weights = ActivationWeights::default();

        let lift = activation(
            0.0,
            Some(&a_veteran(now)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            (0.0..MAX_REINFORCEMENT_DEVIATIONS).contains(&lift),
            "an extreme history lifted {lift} deviations, past the \
             {MAX_REINFORCEMENT_DEVIATIONS} the scale is chosen against"
        );
    }

    /// An equally placed and equally situated candidate scores the same against
    /// any source, however differently that source's distances are spread.
    ///
    /// Named for what it checks rather than for the wider property it was once
    /// called after: `situation` takes no dispersion, so it could not vary with
    /// one even if it were a raw match count. What holds the term to the
    /// semantic term's unit is its ceiling, and
    /// `the_situation_term_is_bounded_by_one_use_at_the_reference_age` is where
    /// that is checked.
    #[test]
    fn an_equally_placed_and_equally_situated_candidate_scores_the_same_against_any_source() {
        use crate::ports::recall::RecallDispersion;

        let now = now();
        let weights = ActivationWeights::default();
        let tight = RecallDispersion::assumed(0.80, 0.05);
        let loose = RecallDispersion::assumed(0.42, 0.30);
        let stands_out_by = 7.5;

        let in_tight = activation(
            tight.deviations_below_median(tight.distance_at(stands_out_by)),
            None,
            1.0,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let in_loose = activation(
            loose.deviations_below_median(loose.distance_at(stands_out_by)),
            None,
            1.0,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            (in_tight - in_loose).abs() < 1e-9,
            "an equally-placed, equally-situated candidate scored {in_tight} against one \
             source and {in_loose} against another"
        );
    }

    /// Acceptance (#1125): the situation term is bounded, the bound is exactly
    /// what one use at the reference age is worth, and it is therefore counted
    /// in the source's own median absolute deviations rather than in a unit of
    /// its own.
    ///
    /// Both halves. The ceiling holds over every coverage a caller could hand
    /// in, including the ones the type does not rule out - a value past one, a
    /// negative value, and a value that is not a number. And the ceiling is the
    /// stated quantity rather than a number of its own: a full match is worth
    /// what an entry opened a day ago is worth, measured against a real record
    /// rather than against the formula.
    #[test]
    fn the_situation_term_is_bounded_by_one_use_at_the_reference_age() {
        let now = now();
        let weights = ActivationWeights::default();
        let a_day = USE_REFERENCE_AGE_SECONDS as i64;

        let ceiling = weights.situation_lift();
        let one_day_old_use = activation(
            0.0,
            Some(&used(now, &[a_day], 1)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        assert!(
            (ceiling - one_day_old_use).abs() < 1e-9,
            "a full situation match is worth {ceiling}, and one use a day old is worth \
             {one_day_old_use}; the bound is supposed to be that equivalence"
        );

        for coverage in [0.0, 0.25, 0.5, 1.0, 1.5, 1e9, -1.0, f64::NAN, f64::INFINITY] {
            let lift = weights.situation(coverage);
            assert!(
                (0.0..=ceiling).contains(&lift),
                "a coverage of {coverage} lifted {lift}, outside the 0 to {ceiling} the term \
                 is bounded to"
            );
        }
    }

    /// Acceptance (#1125): the bound is a stated scale rather than a value
    /// fitted to one store.
    ///
    /// What separates the two, testably, is that a scale carries no unit of its
    /// own. [`USE_REFERENCE_AGE_SECONDS`] is a genuine unit normalization -
    /// restate the use log's ages in another unit and every term is unchanged -
    /// and [`DEFAULT_USE_LIFT`] is not. The situation lift is on the first side
    /// of that line: it is unmoved by the decay exponent, which is what decides
    /// how the ages are counted, and it introduces no coefficient beyond the
    /// `use_lift` a deployment already fits from its own use log.
    #[test]
    fn the_situation_bound_is_a_scale_and_carries_no_unit_of_its_own() {
        for decay in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for use_lift in [0.2, DEFAULT_USE_LIFT, 1.3] {
                let weights = ActivationWeights {
                    use_lift,
                    use_score: UseScoreWeights {
                        decay,
                        ..UseScoreWeights::default()
                    },
                };
                let expected = use_lift * std::f64::consts::LN_2;
                assert!(
                    (weights.situation_lift() - expected).abs() < 1e-9,
                    "at decay {decay} and lift {use_lift} the situation bound was {}, and a \
                     bound that moves with how ages are counted is a fit rather than a scale",
                    weights.situation_lift()
                );
            }
        }
    }

    /// Acceptance (#1125): an entry seen in the present situation is ranked
    /// above an equally similar entry that was not.
    ///
    /// Equal in every other respect - the same distance, and no use history on
    /// either - so the situation is the only thing that separates them.
    #[test]
    fn an_entry_matching_the_present_situation_outranks_an_equally_similar_one() {
        let now = now();
        let weights = ActivationWeights::default();

        let recurs_here = activation(
            7.4,
            None,
            1.0,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let written_elsewhere = activation(
            7.4,
            None,
            0.0,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            recurs_here > written_elsewhere,
            "the entry that recurs here scored {recurs_here} and the one that does not scored \
             {written_elsewhere}"
        );
    }

    /// Acceptance (#1125): with no situation sources connected the score is
    /// exactly the score this module produced before the term existed.
    ///
    /// Not "close to" and not "usually": the same number, over the whole range
    /// of semantic signals and over both an entry with a history and one
    /// without.
    #[test]
    fn with_no_situation_connected_the_score_is_the_pre_change_score() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[60, 6_000], 2);

        for semantic in [0.0, 6.8, 7.3, 11.4, -3.0] {
            let cold = activation(
                semantic,
                None,
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights,
            );
            assert_eq!(
                cold, semantic,
                "an entry with no history and no situation must score its semantic signal alone"
            );

            let used_only = activation(
                semantic,
                Some(&record),
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights,
            );
            assert_eq!(
                used_only,
                semantic + weights.reinforcement(record.use_sum(now, &weights.use_score)),
                "with no cue the score must be the semantic term plus the use log, and nothing \
                 else"
            );
        }
    }

    // --- Salience (#1127) ----------------------------------------------------

    /// An equally placed and equally salient candidate scores the same against
    /// any source, however differently that source's distances are spread.
    ///
    /// Narrower than "the term is counted in deviations", and named for what it
    /// checks: `salience` takes no dispersion, so it could not vary with one
    /// even if it were a raw count. What holds the term to the semantic term's
    /// unit is its ceiling, and
    /// `the_salience_term_is_bounded_by_one_use_at_the_reference_age` is where
    /// that is checked - a term returning `share * 100` passes this test and
    /// fails that one.
    #[test]
    fn an_equally_placed_and_equally_salient_candidate_scores_the_same_against_any_source() {
        use crate::ports::recall::RecallDispersion;

        let now = now();
        let weights = ActivationWeights::default();
        let tight = RecallDispersion::assumed(0.80, 0.05);
        let loose = RecallDispersion::assumed(0.42, 0.30);
        let stands_out_by = 7.5;

        let in_tight = activation(
            tight.deviations_below_median(tight.distance_at(stands_out_by)),
            None,
            NO_SITUATION,
            1.0,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let in_loose = activation(
            loose.deviations_below_median(loose.distance_at(stands_out_by)),
            None,
            NO_SITUATION,
            1.0,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            (in_tight - in_loose).abs() < 1e-9,
            "an equally-placed, equally-salient candidate scored {in_tight} against one source \
             and {in_loose} against another"
        );
    }

    /// Acceptance (#1127): the salience term's size does not grow with how many
    /// signals a deployment can detect.
    ///
    /// The term is a fixed lift spent against a share, and the share is a
    /// ratio, so what every signal is worth on its own sums to exactly one full
    /// reading. A sixth signal takes from the five rather than adding a sixth
    /// part, and the ceiling is the same number it was.
    #[test]
    fn the_salience_signals_divide_one_fixed_lift_rather_than_each_adding_one() {
        use crate::domain::salience::{SalienceReading, SalienceSignal};

        let weights = ActivationWeights::default();
        let ceiling = weights.salience_lift();

        let each: f64 = SalienceSignal::ALL
            .iter()
            .map(|signal| weights.salience(SalienceReading::of([*signal]).share()))
            .sum();
        assert!(
            (each - ceiling).abs() < 1e-9,
            "the signals are worth {each} deviations between them against a ceiling of \
             {ceiling}; a term whose parts do not sum to its ceiling grows with how many parts \
             there are"
        );
    }

    /// Acceptance (#1127): the salience term is bounded, the bound is exactly
    /// what one use at the reference age is worth, and it is therefore counted
    /// in the source's own median absolute deviations rather than in a unit of
    /// its own.
    ///
    /// Both halves. The ceiling holds over every share a caller could hand in,
    /// including the ones the type does not rule out - a value past one, a
    /// negative value, and a value that is not a number. And the ceiling is the
    /// stated quantity rather than a number of its own: a full reading is worth
    /// what an entry opened a day ago is worth, measured against a real record
    /// rather than against the formula.
    #[test]
    fn the_salience_term_is_bounded_by_one_use_at_the_reference_age() {
        let now = now();
        let weights = ActivationWeights::default();
        let a_day = USE_REFERENCE_AGE_SECONDS as i64;

        let ceiling = weights.salience_lift();
        let one_day_old_use = activation(
            0.0,
            Some(&used(now, &[a_day], 1)),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        assert!(
            (ceiling - one_day_old_use).abs() < 1e-9,
            "a full salience reading is worth {ceiling}, and one use a day old is worth \
             {one_day_old_use}; the bound is supposed to be that equivalence"
        );

        for share in [0.0, 0.25, 0.5, 1.0, 1.5, 1e9, -1.0, f64::NAN, f64::INFINITY] {
            let lift = weights.salience(share);
            assert!(
                (0.0..=ceiling).contains(&lift),
                "a share of {share} lifted {lift}, outside the 0 to {ceiling} the term is \
                 bounded to"
            );
        }
    }

    /// Acceptance (#1127): the bound is a stated scale rather than a value
    /// fitted to one store.
    ///
    /// What separates the two, testably, is that a scale carries no unit of its
    /// own. [`USE_REFERENCE_AGE_SECONDS`] is a genuine unit normalization -
    /// restate the use log's ages in another unit and every term is unchanged -
    /// and [`DEFAULT_USE_LIFT`] is not. The salience lift is on the first side
    /// of that line: it is unmoved by the decay exponent, which is what decides
    /// how the ages are counted, and it introduces no coefficient beyond the
    /// `use_lift` a deployment already fits from its own use log.
    #[test]
    fn the_salience_bound_is_a_scale_and_carries_no_unit_of_its_own() {
        for decay in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for use_lift in [0.2, DEFAULT_USE_LIFT, 1.3] {
                let weights = ActivationWeights {
                    use_lift,
                    use_score: UseScoreWeights {
                        decay,
                        ..UseScoreWeights::default()
                    },
                };
                let expected = use_lift * std::f64::consts::LN_2;
                assert!(
                    (weights.salience_lift() - expected).abs() < 1e-9,
                    "at decay {decay} and lift {use_lift} the salience bound was {}, and a bound \
                     that moves with how ages are counted is a fit rather than a scale",
                    weights.salience_lift()
                );
                assert_eq!(
                    weights.salience_lift(),
                    weights.situation_lift(),
                    "both cheap signals are spent against one reference use, so a change to one \
                     must move the other"
                );
            }
        }
    }

    /// Acceptance (#1127): a salient entry is ranked above an equally similar
    /// entry that is not.
    ///
    /// Equal in every other respect - the same distance, no use history on
    /// either, and no situation - so salience is the only thing that separates
    /// them.
    #[test]
    fn a_salient_entry_outranks_an_equally_similar_one_that_is_not() {
        let now = now();
        let weights = ActivationWeights::default();

        let salient = activation(
            7.4,
            None,
            NO_SITUATION,
            1.0,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let plain = activation(
            7.4,
            None,
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        assert!(
            salient > plain,
            "the salient entry scored {salient} and the plain one scored {plain}"
        );
    }

    /// Acceptance (#1127): an entry no detector says anything about scores
    /// exactly what it scored before the term existed.
    ///
    /// Not "close to" and not "usually": the same number, over the whole range
    /// of semantic signals and over both an entry with a history and one
    /// without. This is what makes salience a term and never a gate - a fact
    /// nothing marks salient is stored, retrieved and ranked as it always was.
    #[test]
    fn an_entry_with_no_salience_signal_scores_exactly_what_it_scored_before_the_term_existed() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[60, 6_000], 2);

        for semantic in [0.0, 6.8, 7.3, 11.4, -3.0] {
            assert_eq!(
                activation(
                    semantic,
                    None,
                    NO_SITUATION,
                    NO_SALIENCE,
                    LexicalMatch::NONE,
                    now,
                    &weights
                ),
                semantic
            );
            assert_eq!(
                activation(
                    semantic,
                    Some(&record),
                    NO_SITUATION,
                    NO_SALIENCE,
                    LexicalMatch::NONE,
                    now,
                    &weights,
                ),
                semantic + weights.reinforcement(record.use_sum(now, &weights.use_score))
            );
        }
    }

    /// What the two cheap terms do **together**, which neither term's own test
    /// covers and which is not what either alone can do.
    ///
    /// Each is bounded by one reference use, about a third of a deviation, and
    /// that is under the half a deviation between the bar and the weakest hit
    /// the measured prompts produced - so neither alone takes the top line from
    /// any measured best match. Both at once reach about seven tenths and take
    /// it from the weakest. Pinned in both directions, because a reader who met
    /// only the two single-term tests would carry away a bound that is half the
    /// real one.
    ///
    /// The case where they win is the weakly cued prompt, which is when the
    /// non-semantic signals are supposed to lead (#1123). The case where they
    /// lose is every prompt that named something the store holds.
    #[test]
    fn the_two_cheap_terms_together_reach_twice_what_either_reaches_alone() {
        let now = now();
        let weights = ActivationWeights::default();
        let one = weights.reference_use_lift();

        let both_at_the_bar = activation(BAR, None, 1.0, 1.0, LexicalMatch::NONE, now, &weights);
        assert!(
            (both_at_the_bar - BAR - 2.0 * one).abs() < 1e-9,
            "a fully situated, fully salient candidate at the bar scored {both_at_the_bar}, and \
             the two terms are supposed to be worth {one} each"
        );

        let weakest = *MEASURED_HITS.last().expect("the corpus is not empty");
        assert!(
            weakest - BAR < 2.0 * one,
            "precondition: the weakest measured hit leads the bar by {}, which the two terms \
             together must be able to close",
            weakest - BAR
        );
        assert!(
            both_at_the_bar
                > activation(
                    weakest,
                    None,
                    NO_SITUATION,
                    NO_SALIENCE,
                    LexicalMatch::NONE,
                    now,
                    &weights
                ),
            "on the most weakly cued prompt the corpus holds, an entry that recurs here and \
             carries every salience signal is supposed to lead"
        );

        // And every lead wider than the two of them stands, over the measured
        // corpus rather than at one favourable point.
        let wide: Vec<f64> = MEASURED_HITS
            .iter()
            .copied()
            .filter(|hit| hit - BAR > 2.0 * one)
            .collect();
        assert!(
            wide.len() >= 3,
            "the corpus must hold several prompts that lead by more than both terms: {wide:?}"
        );
        for best in wide {
            assert!(
                both_at_the_bar
                    < activation(
                        best,
                        None,
                        NO_SITUATION,
                        NO_SALIENCE,
                        LexicalMatch::NONE,
                        now,
                        &weights
                    ),
                "a best match at {best} deviations must keep its line against both cheap terms"
            );
        }
    }

    /// An entry no detector reads forfeits the lift; it never subtracts.
    ///
    /// Most of what is worth keeping names no deadline, no money and nobody it
    /// was promised to, so absence must not be evidence against an entry.
    #[test]
    fn an_unsalient_entry_forfeits_the_lift_rather_than_being_pushed_below_the_rest() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[600], 1);

        let without = activation(
            7.0,
            Some(&record),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        assert_eq!(
            activation(
                7.0,
                Some(&record),
                NO_SITUATION,
                0.0,
                LexicalMatch::NONE,
                now,
                &weights
            ),
            without
        );
        assert!(
            activation(
                7.0,
                Some(&record),
                NO_SITUATION,
                0.3,
                LexicalMatch::NONE,
                now,
                &weights
            ) > without
        );
    }

    /// A mismatched situation forfeits the lift; it never subtracts.
    ///
    /// An entry cannot be pushed below what its own distance and history earned
    /// it just because it was first seen somewhere else.
    #[test]
    fn a_mismatched_situation_forfeits_the_lift_rather_than_subtracting() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[600], 1);

        let without = activation(
            7.0,
            Some(&record),
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        assert_eq!(
            activation(
                7.0,
                Some(&record),
                0.0,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights
            ),
            without
        );
        assert!(
            activation(
                7.0,
                Some(&record),
                0.3,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights
            ) > without
        );
    }

    /// The situation cannot overturn a semantic lead, which is the other half of
    /// what the bound is for.
    ///
    /// Over every distance the measured prompts reached that leads the bar by
    /// more than the situation lift, a fully-situated candidate sitting at the
    /// bar stays below a cold best match.
    #[test]
    fn a_lead_wider_than_the_situation_lift_cannot_be_closed_by_the_situation() {
        let now = now();
        let weights = ActivationWeights::default();
        let lift = weights.situation_lift();
        let at_the_bar = activation(
            BAR,
            None,
            1.0,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );

        let wide: Vec<f64> = MEASURED_HITS
            .iter()
            .copied()
            .filter(|hit| hit - BAR > lift)
            .collect();
        assert!(
            wide.len() >= 3,
            "the measured corpus must hold several prompts that lead by more than the lift, \
             or this proves nothing: {wide:?}"
        );
        for best in wide {
            assert!(
                at_the_bar
                    < activation(
                        best,
                        None,
                        NO_SITUATION,
                        NO_SALIENCE,
                        LexicalMatch::NONE,
                        now,
                        &weights
                    ),
                "a fully-situated candidate at the bar scored {at_the_bar} against a cold best \
                 match at {best} deviations"
            );
        }
    }

    // --- The query's own words (#1239) ---------------------------------------

    /// A store whose nearest row for this query stands `spread` deviations
    /// above its furthest, which is what a full lexical match is worth.
    fn a_spread_of(spread: f64) -> LexicalMatch {
        LexicalMatch { share: 1.0, spread }
    }

    /// Acceptance (#1239): a row that carries the query's own words better than
    /// anything else in the store leads a row that is merely nearer.
    ///
    /// The case the term exists for, in the arithmetic that decides it. A store
    /// whose nearest row stands four deviations above its furthest puts a
    /// middling row - the one an exact-token query finds and an embedding does
    /// not - above the nearest row, because carrying the words is worth that
    /// whole spread.
    #[test]
    fn a_full_lexical_match_leads_a_row_that_is_merely_nearest() {
        let now = now();
        let weights = ActivationWeights::default();
        let spread = 4.0;

        let nearest = activation(
            2.5,
            None,
            NO_SITUATION,
            NO_SALIENCE,
            LexicalMatch::NONE,
            now,
            &weights,
        );
        let middling_but_named = activation(
            0.0,
            None,
            NO_SITUATION,
            NO_SALIENCE,
            a_spread_of(spread),
            now,
            &weights,
        );

        assert!(
            middling_but_named > nearest,
            "a row the query names exactly scored {middling_but_named} against the nearest row \
             at {nearest}, and the words are supposed to be worth the store's whole spread"
        );
    }

    /// Acceptance (#1239): a row the query's words did not reach is not lifted,
    /// however far this source's rows are spread.
    ///
    /// The negative of the test above, and the property that makes the term
    /// safe to add: a search over a store with no full-text hit ranks exactly
    /// as it ranked before the term existed.
    #[test]
    fn a_row_the_querys_words_did_not_reach_is_not_lifted() {
        let now = now();
        let weights = ActivationWeights::default();
        let record = used(now, &[600], 1);

        for semantic in [0.0, 6.8, 7.3, 11.4, -3.0] {
            let without = activation(
                semantic,
                Some(&record),
                NO_SITUATION,
                NO_SALIENCE,
                LexicalMatch::NONE,
                now,
                &weights,
            );
            assert_eq!(
                without,
                semantic + weights.reinforcement(record.use_sum(now, &weights.use_score)),
                "a candidate no full-text arm returned must score what it scored before the \
                 term existed"
            );
            // A spread as wide as any store could state, and a share of nothing:
            // the lift is the product, so it is still nothing.
            assert_eq!(
                activation(
                    semantic,
                    Some(&record),
                    NO_SITUATION,
                    NO_SALIENCE,
                    LexicalMatch {
                        share: NO_LEXICAL,
                        spread: 20.0
                    },
                    now,
                    &weights,
                ),
                without
            );
        }
    }

    /// Acceptance (#1239): the term is bounded by this query's own spread, and
    /// it is that spread rather than a number of its own.
    ///
    /// The ceiling holds over every share a caller could hand in, including the
    /// ones the type does not rule out - a value past one, a negative value,
    /// and a value that is not a number.
    #[test]
    fn the_lexical_term_is_bounded_by_the_sources_own_spread() {
        let weights = ActivationWeights::default();

        for spread in [0.0, 0.5, 4.0, 13.0] {
            for share in [0.0, 0.25, 0.5, 1.0, 1.5, 1e9, -1.0, f64::NAN, f64::INFINITY] {
                let lift = weights.lexical(LexicalMatch { share, spread });
                assert!(
                    (0.0..=spread).contains(&lift),
                    "a share of {share} against a spread of {spread} lifted {lift}, outside \
                     the 0 to {spread} the term is bounded to"
                );
            }
        }
    }

    /// Acceptance (#1239): the bound is a stated scale rather than a value
    /// fitted to one store.
    ///
    /// What separates the two, testably, is that a scale carries no coefficient
    /// of its own. The lexical lift is the source's own spread multiplied by a
    /// ratio, so it is unmoved by every weight a deployment may fit - which is
    /// what the two cheap signals cannot say, since both follow `use_lift`.
    #[test]
    fn the_lexical_bound_is_a_scale_and_introduces_no_coefficient() {
        for decay in [0.1, 0.5, 0.9] {
            for use_lift in [0.2, DEFAULT_USE_LIFT, 1.3] {
                let weights = ActivationWeights {
                    use_lift,
                    use_score: UseScoreWeights {
                        decay,
                        ..UseScoreWeights::default()
                    },
                };
                assert!(
                    (weights.lexical(a_spread_of(4.0)) - 4.0).abs() < 1e-9,
                    "at decay {decay} and lift {use_lift} a full lexical match was worth {}, \
                     and a bound that moves with a weight somebody fitted is a fit rather than \
                     a scale",
                    weights.lexical(a_spread_of(4.0))
                );
            }
        }
    }

    /// A partial match is worth its share of the spread, so a row that carries
    /// some of the query's words does not rank as though it carried all of
    /// them.
    #[test]
    fn a_partial_lexical_match_is_worth_its_share_of_the_spread() {
        let weights = ActivationWeights::default();
        let spread = 4.0;

        let full = weights.lexical(LexicalMatch { share: 1.0, spread });
        let half = weights.lexical(LexicalMatch { share: 0.5, spread });

        assert!((full - spread).abs() < 1e-9);
        assert!((half - spread / 2.0).abs() < 1e-9);
    }

    /// A source with no spread to state - one whose rows all sit at one
    /// distance - lifts nothing, rather than dividing by nothing or lifting
    /// everything.
    #[test]
    fn a_source_with_no_spread_lifts_no_lexical_match() {
        let weights = ActivationWeights::default();

        assert_eq!(weights.lexical(a_spread_of(0.0)), 0.0);
        assert_eq!(weights.lexical(a_spread_of(-1.0)), 0.0);
        assert_eq!(weights.lexical(a_spread_of(f64::NAN)), 0.0);
        assert_eq!(weights.lexical(a_spread_of(f64::INFINITY)), 0.0);
    }

    /// Acceptance (#1123): scoring a corpus far larger than any one lookup reads
    /// costs a fraction of the budget the lookup already has.
    ///
    /// The recall scan is held to
    /// `RECALL_SCAN_STATEMENT_TIMEOUT` (four seconds) and reads at most
    /// `RECALL_ENTRY_SCAN_LIMIT` rows, so ten thousand is two hundred times what
    /// a real lookup scores. The bound below is generous enough that a loaded
    /// build machine does not fail it and tight enough that a score which became
    /// a database round trip would.
    #[test]
    fn scoring_ten_thousand_entries_stays_inside_the_search_latency_budget() {
        let now = now();
        let weights = ActivationWeights::default();
        let corpus: Vec<KnowledgeUseRecord> = (0..10_000)
            .map(|i| evenly_over(now, 1 + (i % 40), YEAR))
            .collect();

        let started = std::time::Instant::now();
        let total: f64 = corpus
            .iter()
            .enumerate()
            .map(|(i, record)| {
                activation(
                    7.0 + (i % 5) as f64,
                    Some(record),
                    NO_SITUATION,
                    NO_SALIENCE,
                    LexicalMatch::NONE,
                    now,
                    &weights,
                )
            })
            .sum();
        let spent = started.elapsed();

        assert!(total.is_finite(), "every score must be a number");
        assert!(
            spent < std::time::Duration::from_millis(500),
            "scoring ten thousand entries spent {spent:?}, which is a real share of the four \
             seconds the scan itself has"
        );
    }
}
