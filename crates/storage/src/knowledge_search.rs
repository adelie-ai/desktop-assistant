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

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::activation::{
    ActivationWeights, LexicalMatch, NO_SITUATION, activation,
};
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::domain::salience::{SalienceReading, SalienceSource};
use desktop_assistant_core::ports::recall::RecallDispersion;

/// One row the hybrid search admitted, and what the store could measure about
/// it.
#[derive(Debug, Clone)]
pub(crate) struct SearchCandidate {
    pub entry: KnowledgeEntry,
    /// The cosine distance that measured it, or `None` for a row the full-text
    /// arm admitted and the vector arm cannot compare - no stored vector, or
    /// one from another model.
    pub distance: Option<f64>,
    /// Where this row stands among the rows the query's own words reached, in
    /// `[0, 1]`, and [`NO_LEXICAL`] for a row those words did not reach.
    pub lexical_share: f64,
}

/// Order one search page, best first, and cut it to `limit`.
///
/// **Measured rows first, ranked by activation.** Each is scored by
/// [`activation`] over the semantic term its own source states - how many of
/// that store's median absolute deviations below the store's median this
/// query put it - plus what the use log knows about it and what its own text
/// says about how salient it is. Nothing here is a rank: the distance survives
/// into the score, which is the whole point of the change.
///
/// **The query's own words count too** (#1239). A candidate the full-text arm
/// returned carries a share of this query's own best lexical match, and that
/// share buys it a share of the spread the source's own distances have -
/// [`ActivationWeights::lexical`] states the equivalence and why the spread is
/// the right scale. It is what lets a row an exact-token query finds lead a row
/// that is merely nearer, which is the whole reason the term exists.
///
/// **Then the rows the store could not measure, in the order it gave them.**
/// Such a row carries no distance, so there is no dimensionless term for the
/// score to add and no honest place for it among the measured ones. Standing in
/// a fixed value would say it is as good as a row at the store's median, which
/// is a claim nobody measured; dropping it would hide an entry written since
/// the last embedding backfill. So it keeps the database's own `ts_rank_cd`
/// order and follows - which is the same rule
/// [`RecallRelevance::LexicalMatch`](desktop_assistant_core::ports::recall::RecallRelevance::LexicalMatch)
/// states for a lexical candidate, applied to the one caller that sees both
/// kinds at once.
///
/// `situation` is not read here, and the term is
/// [`NO_SITUATION`] for every candidate. The cue is a
/// property of the whole store measured per turn, and the block already pays
/// for it once a turn; a search runs inside a turn that is already going, and
/// may run several times, so paying for it again per call is a cost this change
/// does not take on. #1240 tracks it.
///
/// `spread` is how many of the source's own deviations separate its nearest row
/// from its furthest for this query, which is what a full lexical match is
/// worth. The scan states it; zero where the source stated none, which leaves
/// every lexical term at nothing.
///
/// The scan answers one row per entry, so nothing here deduplicates: a row both
/// arms admitted arrives once, carrying the distance one measured and the share
/// the other read.
///
/// `records` may be empty: a use log that could not be read costs the order and
/// never the page, exactly as it does on the recall path.
///
/// **This is the second implementation of the ranking policy**, beside
/// `core::recall`'s own, and the two already differ on the mixed set: that one
/// refuses to rank at all where some candidates carry a distance and some do
/// not, because for its caller a mixed set means an adapter fused two modes.
/// Here a mixed set is what the two arms produce by construction, so refusing
/// would turn the ranking off on every call. Both read one `activation`, so the
/// score has one definition; nothing holds the policy or the inputs together,
/// and #1244 is where that is fixed.
pub(crate) fn rank_page(
    candidates: Vec<SearchCandidate>,
    dispersion: RecallDispersion,
    spread: f64,
    records: &HashMap<String, KnowledgeUseRecord>,
    now: DateTime<Utc>,
    limit: usize,
) -> Vec<KnowledgeEntry> {
    let weights = ActivationWeights::default();
    let mut measured: Vec<(f64, KnowledgeEntry)> = Vec::new();
    let mut unmeasured: Vec<KnowledgeEntry> = Vec::new();

    for candidate in candidates {
        let lexical = LexicalMatch {
            share: candidate.lexical_share,
            spread,
        };
        let Some(distance) = candidate.distance else {
            // A row nothing measured still carries the words that found it, but
            // there is no semantic term for the lift to be added to - a lift on
            // its own would place it by a number nobody measured. It keeps the
            // order the database ranked it in, which is that lexical order.
            unmeasured.push(candidate.entry);
            continue;
        };
        let score = activation(
            dispersion.deviations_below_median(distance),
            records.get(&candidate.entry.id),
            NO_SITUATION,
            SalienceReading::read(&SalienceSource::of(&candidate.entry)).share(),
            lexical,
            now,
            &weights,
        );
        measured.push((score, candidate.entry));
    }

    // `total_cmp` rather than `partial_cmp`, so the comparator is a total order
    // and the sort cannot depend on which pair it happened to visit first. The
    // sort is stable, so two candidates that score identically keep the order
    // the scan gave them, which is nearest first.
    measured.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut page: Vec<KnowledgeEntry> = measured.into_iter().map(|(_, entry)| entry).collect();
    page.append(&mut unmeasured);
    page.truncate(limit);
    page
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
    WITH d AS (
         SELECT id, MIN(chunk <=> $1) AS distance
         FROM knowledge_base, unnest(embedding) AS chunk
         WHERE user_id = $6
           AND deleted_at IS NULL
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
     )
     SELECT kb.id, kb.content, kb.tags, kb.metadata, kb.created_at, kb.updated_at,
            kb.source, kb.summary,
            a.distance, a.lexical_share,
            s.median, s.rows_read, s.deviation, s.nearest, s.furthest
     FROM merged a
     JOIN knowledge_base kb
       ON kb.id = a.id AND kb.user_id = $6 AND kb.deleted_at IS NULL
     CROSS JOIN s
     ORDER BY a.arm, a.seat";

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::knowledge_use::RECENT_USE_WINDOW;

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
            lexical_share: 0.0,
        }
    }

    /// A row the query's own words reached as well as the vector arm did.
    fn named(id: &str, distance: f64, share: f64) -> SearchCandidate {
        SearchCandidate {
            entry: an_entry(id),
            distance: Some(distance),
            lexical_share: share,
        }
    }

    fn lexical(id: &str) -> SearchCandidate {
        SearchCandidate {
            entry: an_entry(id),
            distance: None,
            lexical_share: 1.0,
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

    fn log(records: Vec<KnowledgeUseRecord>) -> HashMap<String, KnowledgeUseRecord> {
        records
            .into_iter()
            .map(|r| (r.entry_id.clone(), r))
            .collect()
    }

    fn ids(page: &[KnowledgeEntry]) -> Vec<&str> {
        page.iter().map(|e| e.id.as_str()).collect()
    }

    /// A store whose middling row sits at 0.80 and whose distances vary by 0.05.
    fn a_store() -> RecallDispersion {
        RecallDispersion::assumed(0.80, 0.05)
    }

    /// What a test passes where no candidate carries the query's words, so the
    /// lexical term is worth nothing whatever the spread would have been.
    const NO_SPREAD: f64 = 0.0;

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
        let records = log(vec![used("used", 12, 600)]);

        let page = rank_page(
            vec![measured("nearest", 0.50), measured("used", 0.52)],
            a_store(),
            NO_SPREAD,
            &records,
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
        let records = log(vec![used("used", 12, 600)]);
        let candidates = vec![measured("nearest", 0.50), measured("used", 0.70)];

        let tight = rank_page(
            candidates.clone(),
            RecallDispersion::assumed(0.80, 0.05),
            NO_SPREAD,
            &records,
            now(),
            10,
        );
        let loose = rank_page(
            candidates,
            RecallDispersion::assumed(0.80, 3.0),
            NO_SPREAD,
            &records,
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
            NO_SPREAD,
            &HashMap::new(),
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
            NO_SPREAD,
            &HashMap::new(),
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["first", "second", "third"]);
    }

    /// The cut to the caller's page size happens after the ranking, so a
    /// candidate activation lifts into the page is on it.
    #[test]
    fn the_page_is_cut_to_the_limit_after_ranking_rather_than_before() {
        let records = log(vec![used("used", 12, 600)]);

        let page = rank_page(
            vec![
                measured("nearest", 0.50),
                measured("second", 0.51),
                measured("used", 0.53),
            ],
            a_store(),
            NO_SPREAD,
            &records,
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
            NO_SPREAD,
            &HashMap::new(),
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["near", "far"]);
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
            vec![measured("nearest", 0.70), named("named", 0.80, 1.0)],
            a_store(),
            spread,
            &HashMap::new(),
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
            vec![measured("nearest", 0.70), measured("far", 0.90)],
            a_store(),
            20.0,
            &HashMap::new(),
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
            vec![measured("nearest", 0.70), named("named", 0.80, 1.0)],
            a_store(),
            NO_SPREAD,
            &HashMap::new(),
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
        let candidates = |share| vec![measured("nearest", 0.70), named("named", 0.80, share)];

        assert_eq!(
            ids(&rank_page(
                candidates(1.0),
                a_store(),
                spread,
                &HashMap::new(),
                now(),
                10
            )),
            vec!["named", "nearest"]
        );
        assert_eq!(
            ids(&rank_page(
                candidates(0.1),
                a_store(),
                spread,
                &HashMap::new(),
                now(),
                10
            )),
            vec!["nearest", "named"],
            "a tenth of the words must not buy the whole spread"
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
    fn the_hybrid_scan_selects_what_the_lexical_term_reads() {
        let projection = HYBRID_SEARCH_SQL
            .rsplit("     SELECT kb.id")
            .next()
            .expect("the scan reads its rows after it measures");

        for column in ["a.lexical_share", "s.nearest", "s.furthest"] {
            assert!(
                projection.contains(column),
                "the scan does not select {column}, so the lexical term reads nothing: \
                 \n{HYBRID_SEARCH_SQL}"
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

        for bound in ["user_id = $6", "deleted_at IS NULL", "$2", "$7"] {
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
}
