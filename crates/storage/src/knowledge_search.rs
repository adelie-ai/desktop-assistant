//! The knowledge-base search tool's own read and its ranking (#1167).
//!
//! The tool used to fuse a vector arm and a full-text arm by reciprocal rank.
//! A rank is a position, and a position has already discarded the distance that
//! produced it, so the fused score could not take the one quantity the rest of
//! retrieval ranks by: how far a candidate stands out of its own source,
//! counted in that source's median absolute deviations
//! ([`activation`](desktop_assistant_core::domain::activation)). The tool and
//! the `[Recall]` block therefore ranked by different rules, and a person
//! reading a result could not tell which they had got.
//!
//! ## One ranking rule, held by the type system (#1244)
//!
//! The order is `desktop_assistant_core::ports::recall::rank_by_activation`'s -
//! the same function the `[Recall]` block's arms rank by - and every term it
//! reads comes off `SearchCandidate`'s own `Activatable` implementation.
//! #1167 stated that the tool and the block could not drift because they read
//! one score, and that was already false when written: the two built the
//! score's *arguments* separately, so the page's projection dropping the
//! `source` column, and the situation term being passed as a literal zero, were
//! both differences nothing would have caught. A term added to the score is now
//! a method added to the trait, which does not compile until both callers
//! answer it.
//!
//! The one term the two legitimately differ on is the lexical one, and the
//! difference is stated rather than silent: this caller has a full-text arm and
//! reads it, and a recall lookup uses one mode at a time and answers
//! `LexicalMatch::NONE`.
//!
//! ## What the two arms do now
//!
//! **The arms admit; activation ranks.** That is the same division the block
//! keeps between its bar and its ordering, and it is what the change comes down
//! to:
//!
//! - The vector arm measures every in-scope row this query can be compared
//!   with, states the store's own median and median absolute deviation over
//!   those distances, and admits the nearest of them.
//! - The full-text arm admits every row that carries the query's words, whether
//!   or not the vector arm can compare it. One it can compare arrives with its
//!   distance and is scored like any other; one it cannot - a row written since
//!   the last embedding backfill, or one still stamped with a superseded model -
//!   carries no distance, so it carries no semantic term and no activation
//!   score, and it keeps the order the database ranked it in and follows the
//!   measured rows. See `rank_page`, which is private to this crate.
//!
//! ## The query's own words are a term, not a tiebreak (#1239)
//!
//! Ranking the admitted set on distance alone had one bad consequence, and it
//! was measured rather than argued: on a seeded store of thirty-one rows, a row
//! whose content carried a distinctive identifier sat thirteenth by distance, so
//! a page of five did not hold it. `builtin_knowledge_base_search` is the only
//! text search the model has, so an identifier, a serial number or a quoted
//! phrase an embedding represents poorly could not be found at all.
//!
//! The activation score's own full-text-rank term is what answers that, and this
//! is the caller that supplies its input - a recall lookup uses one mode at a
//! time and so carries no rank. Every candidate the full-text arm returns
//! carries a share of this query's own best lexical match, and that share buys a
//! share of the spread the source's own distances have.
//! [`ActivationWeights::lexical`](desktop_assistant_core::domain::activation::ActivationWeights::lexical)
//! states the equivalence, why the spread is the right scale, and why one
//! reference use is not.
//!
//! Two properties keep it honest, and both are named tests:
//! `knowledge_hybrid_search_puts_an_exactly_named_row_at_the_top_of_the_page`
//! is the measurement above, and
//! `knowledge_hybrid_search_leaves_a_row_the_query_never_names_where_its_distance_puts_it`
//! is its negative - a row with neither a text hit nor a good distance is not
//! lifted, however wide the store is spread.
//!
//! **What is still not a rank.** The share is read from `ts_rank_cd`'s own
//! magnitude against this query's best, never from a position, and the seats the
//! scan carries decide only the order candidates travel in. A rank-shaped
//! tiebreak would reintroduce exactly what this module removed.

use chrono::{DateTime, Utc};
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::activation::{LexicalMatch, NO_SITUATION};
use desktop_assistant_core::domain::knowledge::Disposition;
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::domain::salience::{SalienceReading, SalienceSource};
use desktop_assistant_core::domain::situation::{SituationCue, SituationRecord};
use desktop_assistant_core::ports::recall::{
    Activatable, MixedSet, RecallDispersion, RecallRelevance, rank_by_activation,
};

/// One row the hybrid search admitted, and what the store could measure about
/// it.
///
/// **Every term the activation score reads is a field here, and the
/// [`Activatable`] implementation below is the only place they are read from**
/// (#1244). The `[Recall]` block's candidate ([`RecallEntry`]) carries the same
/// set for the same reason: a term added to the score is a method added to the
/// trait, which is a compile error in both until both answer.
///
/// [`RecallEntry`]: desktop_assistant_core::ports::recall::RecallEntry
#[derive(Debug, Clone)]
pub(crate) struct SearchCandidate {
    pub entry: KnowledgeEntry,
    /// The cosine distance that measured it, or `None` for a row the full-text
    /// arm admitted and the vector arm cannot compare - no stored vector, or
    /// one from another model.
    pub distance: Option<f64>,
    /// What this query's own words found here, and how far this source lets
    /// anything stand out for them (#1239).
    ///
    /// [`LexicalMatch::NONE`] for a row those words did not reach. The spread
    /// is the source's and is the same for every candidate of one page; it
    /// travels per candidate because the term is read through the trait, and a
    /// term read from anywhere but the candidate is a term one caller can
    /// supply and the other forget.
    pub lexical: LexicalMatch,
    /// What the use log knows about this row (#698), or `None` where it knows
    /// nothing or could not be read - both mean the row ranks on its other
    /// terms, which is how every row ranked before the log existed.
    pub use_record: Option<KnowledgeUseRecord>,
    /// The situations this row has been seen in (#1125).
    ///
    /// Empty is an ordinary answer - a row written before any of this was
    /// recorded, a turn with nothing connected, or a read that failed - and
    /// [`SituationCue::coverage`] then answers zero.
    pub situation: SituationRecord,
}

impl Activatable for SearchCandidate {
    /// A row the vector arm measured carries its distance; a row only the
    /// full-text arm reached carries none, which is the same statement
    /// [`RecallRelevance::LexicalMatch`] makes on the recall path.
    fn relevance(&self) -> RecallRelevance {
        self.distance
            .map_or(RecallRelevance::LexicalMatch, RecallRelevance::Distance)
    }

    fn use_record(&self) -> Option<&KnowledgeUseRecord> {
        self.use_record.as_ref()
    }

    fn situation_coverage(&self, cue: Option<&SituationCue>) -> f64 {
        cue.map_or(NO_SITUATION, |cue| cue.coverage(&self.situation))
    }

    /// Read from the row's own stored text and provenance, which the scan
    /// already selects - the same reading the recall path takes of the same
    /// entry, from the same fields.
    fn salience_share(&self) -> f64 {
        SalienceReading::read(&SalienceSource::of(&self.entry)).share()
    }

    /// The one term the two paths legitimately differ on. This caller has a
    /// full-text arm and reads it; a recall lookup has none and answers
    /// [`LexicalMatch::NONE`].
    fn lexical(&self) -> LexicalMatch {
        self.lexical
    }

    /// The row's own stored disposition (#893), read from the same entry the
    /// other terms read.
    fn disposition(&self) -> Disposition {
        self.entry.disposition
    }
}

/// Order one search page, best first, and cut it to `limit`.
///
/// **One ranking rule, held by the type system** (#1244). The order is
/// [`rank_by_activation`]'s, the same function the `[Recall]` block's arms rank
/// by, and every term it reads comes off [`SearchCandidate`]'s own
/// [`Activatable`] implementation. #1167 stated in three places that the tool
/// and the block could not drift because they read one score; that claim had no
/// enforcer and was already false when written - the page's projection dropped
/// the `source` column, so the salience term's `Deliberate` signal could never
/// fire here, and the situation term was passed as a literal zero. Both were
/// *inputs*. A term added to the score is now a method added to the trait, and
/// a method with no default body is a compile error in both implementors until
/// both answer.
///
/// **The mixed-set policy is this caller's, and it is the opposite of the
/// block's.** A search page is mixed by construction - the full-text arm
/// deliberately admits rows the vector arm cannot compare at all - so
/// [`MixedSet::MeasuredFirst`] ranks what was measured and keeps the rest in
/// the order the database gave them. Refusing to rank a mixed set, which is
/// what a recall lookup does because a mixed set there means an adapter fused
/// two modes, would turn the ranking off on every call here.
///
/// Such a row carries no distance, so there is no dimensionless term for a
/// score to add to and no honest place for it among the measured ones. Standing
/// in a fixed value would say it is as good as a row at the store's median,
/// which is a claim nobody measured; dropping it would hide an entry written
/// since the last embedding backfill. So it keeps the database's own
/// `ts_rank_cd` order and follows - the same rule
/// [`RecallRelevance::LexicalMatch`] states for a lexical candidate, applied to
/// the one caller that sees both kinds at once.
///
/// `situation` is the cue the running turn measured against this store, handed
/// down rather than re-read - see
/// [`current_situation_cue`](desktop_assistant_core::ports::knowledge_use::current_situation_cue),
/// which holds why. `None` weights the term at zero, which ranks exactly as
/// this page ranked before the term reached it.
///
/// The scan answers one row per entry, so nothing here deduplicates: a row both
/// arms admitted arrives once, carrying the distance one measured and the share
/// the other read.
pub(crate) fn rank_page(
    candidates: Vec<SearchCandidate>,
    dispersion: RecallDispersion,
    situation: Option<&SituationCue>,
    now: DateTime<Utc>,
    limit: usize,
) -> Vec<KnowledgeEntry> {
    rank_by_activation(
        candidates,
        |candidate| candidate,
        dispersion,
        situation,
        now,
        MixedSet::MeasuredFirst,
    )
    .into_iter()
    .map(|candidate| candidate.entry)
    .take(limit)
    .collect()
}

/// What [`PgKnowledgeBaseStore`](crate::PgKnowledgeBaseStore)'s hybrid search
/// arm reads.
///
/// One scan, four uses, on the same construction the recall scan uses. `d`
/// computes one distance per comparable row and carries nothing else, so the
/// pass that measures the store's spread reads no entry content; `m` takes the
/// median of those distances and `s` the median of each distance's own distance
/// from it. `measured` then admits the nearest rows and `lexical` the rows `d`
/// could not reach at all.
///
/// **The spread is measured over every row the scan could reach**, never over
/// the rows it returns. The returned rows are the near tail, which is the part
/// a cued query moves, so normalizing inside it would inflate every score.
///
/// **The spread cannot be cached.** The median and the deviation are statistics
/// of the distances from *this* query's point, so a query in a dense region of
/// the store has a different distribution from one in a sparse region. They are
/// measured in the pass that ranks, or they describe a geometry nothing here
/// saw.
///
/// **`lexical` admits on the query's words alone, whether or not `d` reached the
/// row.** It left-joins `d`, so a row the vector arm can compare arrives with
/// its distance and is scored like any other, and a row it cannot arrives with
/// none. Excluding the rows `d` reached - which was this query's first shape -
/// made the full-text arm return nothing at all on a store whose rows are
/// embedded, so a row the query names exactly was not merely ranked low but
/// absent from the candidate set, and no later term could ever lift it. The two
/// lists may now hold the same row twice; `rank_page` keeps the first.
///
/// `lexical` is cut to `$5`, the caller's own page size: nothing can put more
/// than a page of them in front of the caller. `measured` is cut to `$3`, which
/// over-fetches, because activation reorders it and a row it lifts has to be in
/// the set to be lifted.
///
/// `merged` folds the two admissions into one row per entry, carrying the
/// distance whichever arm measured it and the share the full-text arm read. The
/// seat it keeps is the vector arm's where that arm admitted the row, so the
/// order rows travel in is the nearest-first order the scan produced and never
/// a lexical position - which is what keeps a rank out of the ordering as well
/// as out of the score. `arm` and `seat` decide only the order the candidates
/// arrive in, which `rank_page` uses to break exact ties and to order the rows
/// nothing could measure.
///
/// `best_rank` is the one extra pass this costs: the share is a ratio against
/// this query's own best full-text match, so the best has to be known before
/// any row's share can be. It rides the same `tsv` index the arm itself does.
/// The alternative - a window function over the arm - would be computed after
/// its `LIMIT` and so would divide by the best of the page rather than the best
/// of the store.
///
/// Both arms carry the whole scope - the user, the live-row predicate, and both
/// tag filters. A predicate present on one arm and missing from the other would
/// make the weaker arm a way around the scope the other enforces. The user and
/// the live-row predicate are repeated on the final join rather than trusted
/// from the arms: an id can only reach it through one of them, but a scope
/// predicate that appears once is a scope predicate one refactor can lose, and
/// this table holds every tenant's knowledge.
///
/// Only the vector arm is model-scoped ($8). A vector of another dimension
/// makes pgvector raise rather than miss, and a table legitimately holds two
/// models' vectors through any reindex. Sameness is decided on the digest half
/// of the `<name>@<digest>` stamp wherever both sides carry one, matching
/// `embedding_backfill::invalidate_stale_embeddings`, so a cosmetic rename does
/// not blank semantic search until the sweep restamps the rows.
/// `split_part(x, '@', 2)` yields '' where there is no '@', so the non-empty
/// test doubles as "both sides carry a digest". The full-text arm is
/// deliberately unscoped, which is what turns a model change into degraded
/// recall rather than content that cannot be found at all.
///
/// ## Disposition admission and resolution (#893)
///
/// **Both arms admit every disposition but `obsolete`.** `active`,
/// `refuted`, `trivial`, `superseded` and `redundant` all reach `d` and
/// `lexical` on the same terms as before this changed - a row's disposition
/// no longer decided whether it could be found, only what happens to the id
/// once it is. `obsolete` is the one exclusion, and `$9` lifts it: a caller
/// that explicitly asked to see dispositioned history gets it, and every
/// other caller does not.
///
/// **`chain` resolves `superseded` and `redundant` through `superseded_by`
/// before anything is shown.** A row admitted under either disposition is
/// never returned under its own id - the search that admitted it wanted
/// what that id says, and what it says now is "ask the row it points to".
/// The walk is a recursive CTE over `superseded_by`, one step per row,
/// bounded at depth 8: `chain.depth < 8` in the recursive term is what makes
/// a cycle terminate rather than hang the query, at the cost of not
/// necessarily reaching a true terminal on a chain that long. Consolidation
/// never writes a chain anywhere near that deep - the bound exists for
/// defense, not for the ordinary case.
///
/// **`terminal` answers every admitted id, not only the resolved ones.** An
/// id whose own disposition is not `superseded`/`redundant` never satisfies
/// the recursive term's `WHERE`, so its own chain has exactly one row and its
/// terminal is itself - which is what lets `resolved`/`final_admitted` treat
/// every id uniformly instead of branching on whether resolution applied.
///
/// **`final_admitted` deduplicates a resolved id against one the arms
/// admitted directly, and the direct admission wins.** Two ids can resolve to
/// the same terminal - a chain's origin and a row the arms separately
/// matched on its own words - and showing both would be the same content
/// twice. `DISTINCT ON (terminal_id)` keeps one row per terminal, and the
/// `ORDER BY` inside it prefers, in order: the id that *is* its own terminal
/// (a direct match needs no resolution to explain its seat), then the
/// vector arm over the full-text arm, then the better seat. **The resolved
/// row inherits the match's seat** in the remaining case - a chain's origin
/// matched and its terminal did not - so a query that only names the old
/// wording still places its successor by how well the old wording matched,
/// not by whatever position the successor's own (unrelated) row would have
/// taken.
///
/// **The final join re-applies both the live-row predicate and the obsolete
/// exclusion to the *terminal* id.** A chain can resolve to a row this
/// caller would otherwise refuse - reaped, or itself `obsolete` - and the
/// terminal's own state is what decides visibility, not the state of the id
/// that led there.
///
/// **A dangling `superseded_by` fails safe.** `superseded_by` carries no
/// foreign key (the same reason a tombstone's own copy does not, see
/// migration 038), so a terminal id can name a row that was hard-deleted
/// after the link was written. The final join is an `INNER JOIN` against
/// `knowledge_base`, so a terminal id that resolves to nothing simply
/// matches no row and the candidate that led to it is dropped - silently,
/// which is the right failure mode for a dangling link, but untested until
/// `a_dangling_superseded_by_target_is_dropped_not_errored` pinned it.
///
/// Every returned row repeats the same three statistics, which is the price of
/// stating them in the same answer as the candidates.
///
/// Held as its own string, and public, for two different checks. Its
/// projection and its arms' scope can be asserted without a database (see
/// `the_hybrid_scan_measures_before_it_reads_any_entry`), and the three
/// statistics it computes cannot: `search_hybrid` consumes them and answers
/// with entries, so nothing downstream can see a median that is wrong rather
/// than merely absent - and a wrong spread silently changes every near tie the
/// reinforcement term exists to settle. `crates/storage/tests/knowledge_hybrid_and_pagination.rs`
/// binds this against a seeded store and asserts the numbers.
pub const HYBRID_SEARCH_SQL: &str = "\
    WITH RECURSIVE d AS (
         SELECT id, MIN(chunk <=> $1) AS distance
         FROM knowledge_base, unnest(embedding) AS chunk
         WHERE user_id = $6
           AND deleted_at IS NULL
           AND (disposition <> 'obsolete' OR $9)
           AND ($2::text[] IS NULL OR tags && $2)
           AND ($7::text[] IS NULL OR NOT (tags && $7))
           AND embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND (embedding_model = $8
                OR (split_part($8, '@', 2) <> ''
                    AND split_part(embedding_model, '@', 2)
                        = split_part($8, '@', 2)))
         GROUP BY id
     ),
     m AS (
         SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY distance) AS median,
                count(*) AS rows_read,
                min(distance) AS nearest,
                max(distance) AS furthest
         FROM d
     ),
     s AS (
         SELECT m.median,
                m.rows_read,
                m.nearest,
                m.furthest,
                (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(d.distance - m.median))
                 FROM d) AS deviation
         FROM m
     ),
     best_rank AS (
         SELECT max(ts_rank_cd(kb.tsv, query)) AS top
         FROM knowledge_base kb
         CROSS JOIN plainto_tsquery('english', $4) AS query
         WHERE kb.user_id = $6
           AND kb.deleted_at IS NULL
           AND (kb.disposition <> 'obsolete' OR $9)
           AND ($2::text[] IS NULL OR kb.tags && $2)
           AND ($7::text[] IS NULL OR NOT (kb.tags && $7))
           AND kb.tsv @@ query
     ),
     measured AS (
         SELECT id,
                distance,
                row_number() OVER (ORDER BY distance, id DESC) AS seat
         FROM d
         ORDER BY distance, id DESC
         LIMIT $3
     ),
     lexical AS (
         SELECT kb.id,
                d.distance,
                CASE WHEN b.top > 0 THEN ts_rank_cd(kb.tsv, query) / b.top ELSE 0 END
                    AS lexical_share,
                row_number() OVER (ORDER BY ts_rank_cd(kb.tsv, query) DESC,
                                            kb.updated_at DESC, kb.id DESC) AS seat
         FROM knowledge_base kb
         CROSS JOIN plainto_tsquery('english', $4) AS query
         CROSS JOIN best_rank b
         LEFT JOIN d ON d.id = kb.id
         WHERE kb.user_id = $6
           AND kb.deleted_at IS NULL
           AND (kb.disposition <> 'obsolete' OR $9)
           AND ($2::text[] IS NULL OR kb.tags && $2)
           AND ($7::text[] IS NULL OR NOT (kb.tags && $7))
           AND kb.tsv @@ query
         ORDER BY ts_rank_cd(kb.tsv, query) DESC, kb.updated_at DESC, kb.id DESC
         LIMIT $5
     ),
     admitted AS (
         SELECT id, distance, 0::float8 AS lexical_share, 0 AS arm, seat FROM measured
         UNION ALL
         SELECT id, distance, lexical_share::float8, 1 AS arm, seat FROM lexical
     ),
     merged AS (
         SELECT id,
                max(distance) AS distance,
                max(lexical_share) AS lexical_share,
                min(arm) AS arm,
                coalesce(min(seat) FILTER (WHERE arm = 0), min(seat)) AS seat
         FROM admitted
         GROUP BY id
     ),
     chain(start_id, current_id, depth) AS (
         SELECT id, id, 0
         FROM merged
         UNION ALL
         SELECT chain.start_id, kb.superseded_by, chain.depth + 1
         FROM chain
         JOIN knowledge_base kb
           ON kb.id = chain.current_id AND kb.user_id = $6 AND kb.deleted_at IS NULL
         WHERE kb.disposition IN ('superseded', 'redundant')
           AND kb.superseded_by IS NOT NULL
           AND chain.depth < 8
     ),
     terminal AS (
         SELECT DISTINCT ON (start_id) start_id, current_id AS terminal_id
         FROM chain
         ORDER BY start_id, depth DESC
     ),
     resolved AS (
         SELECT m.id AS original_id, m.distance, m.lexical_share, m.arm, m.seat,
                t.terminal_id
         FROM merged m
         JOIN terminal t ON t.start_id = m.id
     ),
     final_admitted AS (
         SELECT DISTINCT ON (terminal_id)
                terminal_id AS id, distance, lexical_share, arm, seat
         FROM resolved
         ORDER BY terminal_id,
                  (terminal_id = original_id) DESC,
                  arm,
                  seat
     )
     SELECT kb.id, kb.content, kb.tags, kb.metadata, kb.created_at, kb.updated_at,
            kb.source, kb.summary, kb.disposition,
            a.distance, a.lexical_share,
            s.median, s.rows_read, s.deviation, s.nearest, s.furthest
     FROM final_admitted a
     JOIN knowledge_base kb
       ON kb.id = a.id AND kb.user_id = $6 AND kb.deleted_at IS NULL
       AND (kb.disposition <> 'obsolete' OR $9)
     CROSS JOIN s
     ORDER BY a.arm, a.seat";

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::activation::{ActivationWeights, NO_LEXICAL};
    use desktop_assistant_core::domain::knowledge_use::RECENT_USE_WINDOW;
    use desktop_assistant_core::domain::salience::SOURCE_EXPLICIT;
    use desktop_assistant_core::domain::situation::{FieldFan, Situation, SituationField};
    use desktop_assistant_core::ports::recall::RecallEntry;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-10T12:00:00Z")
            .expect("a fixed clock parses")
            .with_timezone(&Utc)
    }

    fn an_entry(id: &str) -> KnowledgeEntry {
        KnowledgeEntry::new(id, "a stored fact about the deploy window", vec![])
    }

    fn measured(id: &str, distance: f64) -> SearchCandidate {
        SearchCandidate {
            entry: an_entry(id),
            distance: Some(distance),
            lexical: LexicalMatch::NONE,
            use_record: None,
            situation: SituationRecord::new(),
        }
    }

    /// A row the query's own words reached as well as the vector arm did, in a
    /// source whose nearest and furthest rows stand `spread` deviations apart.
    fn named(id: &str, distance: f64, share: f64, spread: f64) -> SearchCandidate {
        SearchCandidate {
            lexical: LexicalMatch { share, spread },
            ..measured(id, distance)
        }
    }

    /// A row the query's words did not reach, in a source spread that wide.
    /// The pair to [`named`]: the spread is stated and the share is nothing.
    fn unnamed(id: &str, distance: f64, spread: f64) -> SearchCandidate {
        named(id, distance, NO_LEXICAL, spread)
    }

    fn lexical(id: &str) -> SearchCandidate {
        SearchCandidate {
            distance: None,
            lexical: LexicalMatch {
                share: 1.0,
                spread: NO_SPREAD,
            },
            ..measured(id, 0.0)
        }
    }

    /// A record of `opens` opens, the newest of them `seconds_ago` old.
    fn used(id: &str, opens: u64, seconds_ago: i64) -> KnowledgeUseRecord {
        let ages: Vec<i64> = (0..opens.min(RECENT_USE_WINDOW as u64))
            .map(|i| seconds_ago + i as i64 * 60)
            .collect();
        KnowledgeUseRecord {
            entry_id: id.to_string(),
            offered_count: opens,
            opened_count: opens,
            marked_count: 0,
            first_seen_at: now()
                - chrono::TimeDelta::seconds(ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(now() - chrono::TimeDelta::seconds(seconds_ago)),
            recent_uses: ages
                .iter()
                .map(|a| now() - chrono::TimeDelta::seconds(*a))
                .collect(),
            marks: Vec::new(),
        }
    }

    /// The same candidate, carrying `opens` opens the newest of which is
    /// `seconds_ago` old.
    fn opened(candidate: SearchCandidate, opens: u64, seconds_ago: i64) -> SearchCandidate {
        SearchCandidate {
            use_record: Some(used(&candidate.entry.id, opens, seconds_ago)),
            ..candidate
        }
    }

    /// The same candidate, carrying `disposition` instead of the
    /// [`Disposition::Active`] every other helper here builds.
    fn disposed(mut candidate: SearchCandidate, disposition: Disposition) -> SearchCandidate {
        candidate.entry.disposition = disposition;
        candidate
    }

    /// The same candidate, having been seen in `situation`.
    fn seen_in(candidate: SearchCandidate, situation: &Situation) -> SearchCandidate {
        let record = situation
            .iter()
            .fold(SituationRecord::new(), |record, (field, value)| {
                record.with(field, value)
            });
        SearchCandidate {
            situation: record,
            ..candidate
        }
    }

    fn ids(page: &[KnowledgeEntry]) -> Vec<&str> {
        page.iter().map(|e| e.id.as_str()).collect()
    }

    /// A store whose middling row sits at 0.80 and whose distances vary by 0.05.
    fn a_store() -> RecallDispersion {
        RecallDispersion::assumed(0.80, 0.05)
    }

    /// What a test states where the source measured no spread, so a full
    /// lexical match is worth nothing.
    const NO_SPREAD: f64 = 0.0;

    /// What a test passes where the turn measured no situation cue, which is
    /// every turn with nothing connected and every turn that ran no recall
    /// lookup.
    const NO_CUE: Option<&SituationCue> = None;

    /// Acceptance (#1167): the page is ordered by the activation score, so what
    /// the use log knows about a candidate can take the top line from a
    /// marginally nearer one nothing has opened.
    ///
    /// Named for exactly that, and no wider: this function has no concept of an
    /// arm or of a rank, so it cannot check that nothing is fused.
    /// `the_hybrid_scan_states_no_reciprocal_rank_fusion` checks the query, and
    /// `knowledge_hybrid_search_orders_by_activation_and_not_by_a_fused_rank`
    /// checks the whole path against a database over a fixture where the two
    /// rules disagree.
    #[test]
    fn a_used_entry_leads_a_marginally_nearer_one_nothing_has_opened() {
        let page = rank_page(
            vec![
                measured("nearest", 0.50),
                opened(measured("used", 0.52), 12, 600),
            ],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(
            ids(&page),
            vec!["used", "nearest"],
            "an entry the work keeps needing must lead a marginally nearer one nothing has \
             opened"
        );
    }

    /// Acceptance (#1167): the semantic term is read against the store's own
    /// spread, so what a given gap in distance is worth depends on how spread
    /// out that store's distances are.
    ///
    /// The same two candidates and the same use history, under two stores. In
    /// the tight store the gap is four of its deviations and no history closes
    /// it; in the loose store the same gap is under a tenth of a deviation and
    /// the history decides. A ranking that read the raw distance could not tell
    /// the two stores apart.
    #[test]
    fn the_semantic_term_is_read_against_the_stores_own_spread() {
        let candidates = vec![
            measured("nearest", 0.50),
            opened(measured("used", 0.70), 12, 600),
        ];

        let tight = rank_page(
            candidates.clone(),
            RecallDispersion::assumed(0.80, 0.05),
            NO_CUE,
            now(),
            10,
        );
        let loose = rank_page(
            candidates,
            RecallDispersion::assumed(0.80, 3.0),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(
            ids(&tight),
            vec!["nearest", "used"],
            "in a store whose rows barely vary, a gap of four deviations stands"
        );
        assert_eq!(
            ids(&loose),
            vec!["used", "nearest"],
            "in a store whose rows vary widely, the same raw gap is nothing and the log decides"
        );
    }

    /// A row the vector arm cannot compare carries no semantic term, so it is
    /// not scored: it keeps the order the database gave it and follows the rows
    /// that were scored.
    ///
    /// Named for exactly that. It does **not** claim such a row is worth less
    /// than a measured one - nothing measured it - only that there is no honest
    /// number that places it among them.
    #[test]
    fn a_row_the_vector_arm_cannot_compare_keeps_the_database_order_and_follows() {
        let page = rank_page(
            vec![
                measured("far", 0.79),
                lexical("lex-first"),
                lexical("lex-second"),
                measured("near", 0.50),
            ],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["near", "far", "lex-first", "lex-second"]);
    }

    /// A page of nothing but unmeasurable rows is the ordinary answer for a
    /// store whose embeddings have not been written yet, and it comes back in
    /// the order the database ranked it.
    #[test]
    fn a_store_with_no_comparable_row_answers_in_the_databases_own_order() {
        let page = rank_page(
            vec![lexical("first"), lexical("second"), lexical("third")],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["first", "second", "third"]);
    }

    /// The cut to the caller's page size happens after the ranking, so a
    /// candidate activation lifts into the page is on it.
    #[test]
    fn the_page_is_cut_to_the_limit_after_ranking_rather_than_before() {
        let page = rank_page(
            vec![
                measured("nearest", 0.50),
                measured("second", 0.51),
                opened(measured("used", 0.53), 12, 600),
            ],
            a_store(),
            NO_CUE,
            now(),
            1,
        );

        assert_eq!(ids(&page), vec!["used"]);
    }

    /// A use log that could not be read costs the order and never the page:
    /// every candidate then ranks on its semantic signal alone, which is how
    /// they all ranked before the log existed.
    #[test]
    fn a_page_with_no_use_records_is_ranked_on_the_semantic_signal_alone() {
        let page = rank_page(
            vec![measured("far", 0.60), measured("near", 0.50)],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["near", "far"]);
    }

    /// Acceptance (#893): with every other term equal, a `trivial` entry
    /// ranks below an `active` one of comparable relevance.
    ///
    /// Both candidates sit at the same distance, so nothing but the
    /// disposition term can separate them - the same "equal but for the one
    /// thing under test" shape `a_search_ranks_an_entry_seen_in_the_present_situation_above_an_equally_similar_one_that_was_not`
    /// uses for the situation term.
    #[test]
    fn a_trivial_entry_ranks_below_an_active_one_of_comparable_relevance() {
        // Arrived in the order the penalty must overturn: `trivial` first,
        // `active` second. A stable sort leaves equal scores in arrival
        // order, so a page that merely preserved this order - the penalty
        // contributing nothing - would put `trivial` first too. Only a
        // penalty that actually fires can put `active` ahead of the
        // candidate that arrived before it.
        let page = rank_page(
            vec![
                disposed(measured("trivial", 0.60), Disposition::Trivial),
                measured("active", 0.60),
            ],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(
            ids(&page),
            vec!["active", "trivial"],
            "a trivial entry of comparable relevance must rank below an active one, even when \
             it arrived first"
        );
    }

    /// The negative half of the test above: two `active` candidates at the
    /// same distance are unaffected by the disposition term at all, so the
    /// penalty above is not merely a tiebreak that happens to favour whoever
    /// arrived first.
    #[test]
    fn two_active_entries_of_equal_relevance_are_unaffected_by_the_disposition_term() {
        let page = rank_page(
            vec![measured("first", 0.60), measured("second", 0.60)],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        // A stable sort leaves equal scores in arrival order - the same
        // property `a_page_with_no_use_records_is_ranked_on_the_semantic_signal_alone`
        // rests on. This test is about the disposition term contributing
        // nothing here, not about which one happens to sort first.
        assert_eq!(ids(&page), vec!["first", "second"]);
    }

    /// Acceptance (#1239): a row the query's own words name leads a row that is
    /// merely nearer, so an exact-token search finds its entry.
    ///
    /// The unit half of the measurement `knowledge_hybrid_search_puts_an_exactly_named_row_at_the_top_of_the_page`
    /// makes against a database. "named" sits at the store's median distance -
    /// a middling row an embedding has no opinion about - and carries the
    /// query's words as exclusively as anything in the store; "nearest" is the
    /// closest row and carries none of them.
    #[test]
    fn a_row_the_querys_words_name_leads_a_row_that_is_merely_nearer() {
        let spread = 4.0;

        let page = rank_page(
            vec![
                unnamed("nearest", 0.70, spread),
                named("named", 0.80, 1.0, spread),
            ],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(
            ids(&page),
            vec!["named", "nearest"],
            "a row the query names must lead one that is merely nearer"
        );
    }

    /// Acceptance (#1239): a row with neither a text hit nor a good distance is
    /// not lifted by this term.
    ///
    /// The negative, and the property that makes the term safe: the lift is a
    /// share of the spread, so a share of nothing is nothing however wide the
    /// source is spread.
    #[test]
    fn a_row_with_no_text_hit_is_not_lifted_however_wide_the_source_is_spread() {
        let far_and_unnamed = rank_page(
            vec![unnamed("nearest", 0.70, 20.0), unnamed("far", 0.90, 20.0)],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(
            ids(&far_and_unnamed),
            vec!["nearest", "far"],
            "a wide spread must lift nobody the query's words did not reach"
        );
    }

    /// A source that stated no spread lifts nothing, so a store too small to
    /// measure ranks exactly as it ranked before the term existed.
    #[test]
    fn a_source_with_no_measured_spread_ranks_as_it_did_before_the_term_existed() {
        let page = rank_page(
            vec![
                unnamed("nearest", 0.70, NO_SPREAD),
                named("named", 0.80, 1.0, NO_SPREAD),
            ],
            a_store(),
            NO_CUE,
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["nearest", "named"]);
    }

    /// A partial text hit buys a partial lift, so a row that carries some of
    /// the query's words does not rank as though it carried all of them.
    #[test]
    fn a_partial_text_hit_does_not_lift_as_far_as_a_full_one() {
        let spread = 4.0;
        let candidates = |share| {
            vec![
                unnamed("nearest", 0.70, spread),
                named("named", 0.80, share, spread),
            ]
        };

        assert_eq!(
            ids(&rank_page(candidates(1.0), a_store(), NO_CUE, now(), 10)),
            vec!["named", "nearest"]
        );
        assert_eq!(
            ids(&rank_page(candidates(0.1), a_store(), NO_CUE, now(), 10)),
            vec!["nearest", "named"],
            "a tenth of the words must not buy the whole spread"
        );
    }

    // --- One ranking rule for the tool and the block (#1244) ----------------

    /// The present situation the tests below rank in.
    fn here_and_now() -> Situation {
        Situation::new()
            .with(SituationField::Host, "workshop")
            .with(SituationField::Weekday, "thursday")
    }

    /// A store of two hundred entries in which each of the cue's values is
    /// carried by a quarter of them, so every field is informative and none
    /// dominates.
    fn a_gradeable_cue(situation: Situation) -> SituationCue {
        let fans = situation
            .iter()
            .map(|(field, _)| {
                (
                    field,
                    FieldFan {
                        population: 200,
                        holding: 50,
                    },
                )
            })
            .collect();
        SituationCue::measured(situation, &fans).expect("two hundred entries is a gradeable store")
    }

    /// An entry a person wrote in a live turn, so the salience term has a
    /// signal to read rather than nothing.
    fn a_deliberate_entry(id: &str) -> KnowledgeEntry {
        KnowledgeEntry {
            source: Some(SOURCE_EXPLICIT.to_string()),
            ..an_entry(id)
        }
    }

    /// The same knowledge entry as the search tool's candidate and as the
    /// block's, carrying the same history and the same situation record.
    fn both_ways(
        entry: KnowledgeEntry,
        distance: f64,
        record: KnowledgeUseRecord,
        situation: &Situation,
    ) -> (RecallEntry, SearchCandidate) {
        let seen = situation
            .iter()
            .fold(SituationRecord::new(), |record, (field, value)| {
                record.with(field, value)
            });
        let block = RecallEntry::new(entry.clone(), RecallRelevance::Distance(distance))
            .with_use_record(Some(record.clone()))
            .with_situation(seen.clone());
        let tool = SearchCandidate {
            entry,
            distance: Some(distance),
            lexical: LexicalMatch::NONE,
            use_record: Some(record),
            situation: seen,
        };
        (block, tool)
    }

    /// Acceptance (#1244): the search tool and the `[Recall]` block read one
    /// entry through the same terms, and rank one pair of entries the same way.
    ///
    /// Named for exactly that and no wider. It checks the **four terms the two
    /// paths must agree on** - the semantic signal, the reinforcement the use
    /// log supplies, the situation coverage and the salience share - and one
    /// pair ordered by each path. It does not check the fifth term: the two
    /// legitimately differ there, because a search has a full-text arm and a
    /// recall lookup uses one mode at a time and has none. Nor is it what holds
    /// the two paths together - [`Activatable`] is, by refusing to compile an
    /// implementor that does not answer every term. This is the second line,
    /// and the reason it cannot be the first is that it only ever checks the
    /// terms somebody thought to list here.
    ///
    /// The fixture exercises every term it compares, asserted rather than
    /// assumed: a fixture where the situation and salience terms were both zero
    /// would pass this test with either path ignoring them.
    #[test]
    fn the_tool_and_the_block_supply_the_same_four_shared_terms_and_rank_the_same_pair_alike() {
        let here = here_and_now();
        let cue = a_gradeable_cue(here.clone());
        let store = a_store();
        let weights = ActivationWeights::default();

        let (block, tool) = both_ways(
            a_deliberate_entry("kb-1"),
            store.distance_at(4.0),
            used("kb-1", 12, 600),
            &here,
        );

        assert_eq!(
            block.relevance().semantic_signal(store),
            tool.relevance().semantic_signal(store),
            "the two paths must read one distance against one source the same way"
        );
        assert_eq!(
            block
                .use_record()
                .map(|record| record.use_sum(now(), &weights.use_score)),
            tool.use_record()
                .map(|record| record.use_sum(now(), &weights.use_score)),
            "the two paths must read one use log the same way"
        );
        assert_eq!(
            block.situation_coverage(Some(&cue)),
            tool.situation_coverage(Some(&cue)),
            "the two paths must read one situation record against one cue the same way"
        );
        assert_eq!(
            block.salience_share(),
            tool.salience_share(),
            "the two paths must read one entry's own text and provenance the same way"
        );
        assert!(
            block.situation_coverage(Some(&cue)) > 0.0 && block.salience_share() > 0.0,
            "the fixture must exercise the two terms a path could ignore for free"
        );

        // And the terms reach the ordering, not merely the trait. The two
        // entries sit at one distance, so only the situation can separate them,
        // and the one that was never seen here arrives first.
        let elsewhere = Situation::new()
            .with(SituationField::Host, "the-road")
            .with(SituationField::Weekday, "sunday");
        let (block_far, tool_far) = both_ways(
            a_deliberate_entry("elsewhere"),
            store.distance_at(4.0),
            used("elsewhere", 12, 600),
            &elsewhere,
        );
        let (block_near, tool_near) = both_ways(
            a_deliberate_entry("here"),
            store.distance_at(4.0),
            used("here", 12, 600),
            &here,
        );

        let block_order = rank_by_activation(
            vec![block_far, block_near],
            |hit| hit,
            store,
            Some(&cue),
            now(),
            MixedSet::Refuse,
        );
        let block_ids: Vec<&str> = block_order
            .iter()
            .map(|hit| hit.entry.id.as_str())
            .collect();
        let page = rank_page(vec![tool_far, tool_near], store, Some(&cue), now(), 10);

        assert_eq!(
            block_ids,
            vec!["here", "elsewhere"],
            "the block ranks the entry seen here first"
        );
        assert_eq!(
            ids(&page),
            block_ids,
            "the tool must order the same pair the same way as the block"
        );
    }

    /// Acceptance (#1244): a search ranks an entry seen in the present
    /// situation above an equally similar one that was not.
    ///
    /// Both sit at one distance, carry no use history and carry the same text,
    /// so the situation is the only term that can separate them - and the one
    /// that was never seen here arrives first, so a stable sort left alone
    /// would keep it there.
    #[test]
    fn a_search_ranks_an_entry_seen_in_the_present_situation_above_an_equally_similar_one_that_was_not()
     {
        let here = here_and_now();
        let cue = a_gradeable_cue(here.clone());

        let page = rank_page(
            vec![
                measured("elsewhere", 0.60),
                seen_in(measured("here", 0.60), &here),
            ],
            a_store(),
            Some(&cue),
            now(),
            10,
        );

        assert_eq!(
            ids(&page),
            vec!["here", "elsewhere"],
            "an entry this situation recurs with must lead an equally similar one it does not"
        );
    }

    /// Acceptance (#1244): a turn that measured no cue ranks the page exactly
    /// as it ranked before the term existed.
    ///
    /// Nothing connected, recall not wired, and a store too small to grade a
    /// cue all arrive here as `None`, and the check is stronger than "the order
    /// did not change": the page of entries that carry situation records is the
    /// same page as one whose entries carry none, so the term cannot be reading
    /// a record through some other route.
    #[test]
    fn a_search_with_no_situation_cue_ranks_the_page_as_it_ranked_before_the_cue() {
        let here = here_and_now();
        let situated = vec![
            measured("elsewhere", 0.60),
            seen_in(measured("here", 0.60), &here),
        ];
        let unrecorded = vec![measured("elsewhere", 0.60), measured("here", 0.60)];

        let with_records = rank_page(situated, a_store(), NO_CUE, now(), 10);
        let without_records = rank_page(unrecorded, a_store(), NO_CUE, now(), 10);

        assert_eq!(
            ids(&with_records),
            ids(&without_records),
            "with no cue, a situation record must not move anything"
        );
        assert_eq!(
            ids(&with_records),
            vec!["elsewhere", "here"],
            "and the page keeps the order the scan gave it"
        );
    }

    /// The columns the term reads reach the scorer.
    ///
    /// The lesson of the `source` column, which the projection dropped and
    /// which silently disabled a salience signal: a term that reads a column
    /// nothing selects is a term that never fires, and no ranking test can see
    /// it. `the_hybrid_scan_selects_what_the_lexical_term_reads` is the same
    /// check on the query.
    #[test]
    fn the_hybrid_scan_selects_every_column_the_shared_terms_read() {
        // The trait holds that both paths *read* the same terms. It cannot hold
        // that both paths *populate* them, and this projection is where the
        // tool's population happens - so a column dropped here is a difference
        // between the two rankings that compiles, passes the cross-check test,
        // and changes the order silently.
        //
        // Both defects found so far were exactly that. `kb.source` was dropped
        // once, which made the salience term's `Deliberate` signal unreachable
        // on this path and scored every deliberately-written entry about 0.07
        // deviations lower from the tool than from the block; it was caught by
        // a reviewer noticing a missing column, not by anything that failed.
        //
        // Widen this list when a term is added to `Activatable`.
        let projection = HYBRID_SEARCH_SQL
            .rsplit("     SELECT kb.id")
            .next()
            .expect("the scan reads its rows after it measures");

        for (column, term) in [
            ("a.lexical_share", "the lexical term"),
            ("s.nearest", "the lexical term's own scale"),
            ("s.furthest", "the lexical term's own scale"),
            ("kb.source", "the salience term's Deliberate signal"),
            ("kb.summary", "the salience reading"),
            ("kb.disposition", "the disposition term"),
        ] {
            assert!(
                projection.contains(column),
                "the scan does not select {column}, so {term} reads nothing on the \
                 tool path while the block still reads it: \n{HYBRID_SEARCH_SQL}"
            );
        }
    }

    /// The scan answers one row per entry, so a row both arms admit is not
    /// scored twice or shown twice.
    #[test]
    fn the_hybrid_scan_answers_one_row_per_entry() {
        assert!(
            HYBRID_SEARCH_SQL.contains("GROUP BY id"),
            "the two arms admit independently, so the scan has to fold them: \
             \n{HYBRID_SEARCH_SQL}"
        );
    }

    /// Acceptance (#1167): the pass that measures the store's spread reads the
    /// geometry and none of the content - one distance per row, and no column
    /// of the entry itself.
    #[test]
    fn the_hybrid_scan_measures_before_it_reads_any_entry() {
        let measured = HYBRID_SEARCH_SQL
            .split("     SELECT kb.id")
            .next()
            .expect("the scan selects the rows it will show after it measures the spread");

        for column in ["content", "metadata", "summary"] {
            assert!(
                !measured.contains(column),
                "the pass that measures the spread reads {column}, which it has no use for"
            );
        }
    }

    /// The scan states no fused rank anywhere. A reciprocal-rank term is what
    /// this change removed, and it is the shape a later edit would reach for
    /// first.
    #[test]
    fn the_hybrid_scan_states_no_reciprocal_rank_fusion() {
        for fused in ["rrf", "1.0 / (60", "FULL OUTER JOIN"] {
            assert!(
                !HYBRID_SEARCH_SQL.contains(fused),
                "the scan still carries {fused}, so a rank is deciding the page: \
                 \n{HYBRID_SEARCH_SQL}"
            );
        }
    }

    /// Both arms carry the whole scope. A predicate on one arm and not the
    /// other would make the weaker arm a way around the scope the other
    /// enforces - and this table holds every tenant's knowledge.
    #[test]
    fn both_arms_of_the_hybrid_scan_carry_the_same_scope() {
        // Bounded at the next CTE, so the slice is the arm alone. Reaching past
        // it would take in the final join, which repeats the same scope - and a
        // test that reads the scope twice cannot see it removed from the arm.
        let lexical = HYBRID_SEARCH_SQL
            .split("lexical AS (")
            .nth(1)
            .and_then(|arm| arm.split("     admitted AS (").next())
            .expect("the scan has a full-text arm");
        let measured = HYBRID_SEARCH_SQL
            .split("     m AS (")
            .next()
            .expect("the scan has a vector arm");

        for bound in [
            "user_id = $6",
            "deleted_at IS NULL",
            "$2",
            "$7",
            "disposition <> 'obsolete' OR $9",
        ] {
            assert!(
                lexical.contains(bound),
                "the full-text arm is not bounded by {bound}: \n{HYBRID_SEARCH_SQL}"
            );
            assert!(
                measured.contains(bound),
                "the vector arm is not bounded by {bound}: \n{HYBRID_SEARCH_SQL}"
            );
        }
    }

    // --- Disposition admission and resolution (#893) ------------------------

    /// The `obsolete` exclusion is stated once per gate a row can reach the
    /// caller through: the vector arm, the full-text arm, the best-match
    /// scale the full-text arm's share is read against, and the final join -
    /// so an obsolete row cannot be admitted in the first place, and a chain
    /// cannot resolve past it either.
    #[test]
    fn the_obsolete_exclusion_guards_every_gate() {
        let occurrences = HYBRID_SEARCH_SQL
            .matches("disposition <> 'obsolete' OR $9")
            .count();
        assert_eq!(
            occurrences, 4,
            "the obsolete exclusion must guard the vector arm, the full-text arm, its \
             best-match scale, and the final join - found {occurrences} of the 4: \
             \n{HYBRID_SEARCH_SQL}"
        );
    }

    /// The recursive resolution is bounded, which is what keeps a cycle from
    /// hanging the query - see `a_superseded_chain_resolves_to_the_terminal_successor_and_a_cycle_cannot_hang_it`
    /// in `crates/storage/tests/knowledge_hybrid_and_pagination.rs` for the
    /// behaviour this bound protects.
    #[test]
    fn the_resolution_chain_is_depth_bounded() {
        assert!(
            HYBRID_SEARCH_SQL.contains("chain.depth < 8"),
            "the recursive resolution has no depth bound, so a cycle in \
             superseded_by would hang the query: \n{HYBRID_SEARCH_SQL}"
        );
    }
}
