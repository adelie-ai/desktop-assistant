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
//! - The full-text arm admits rows the vector arm cannot compare at all - a row
//!   written since the last embedding backfill, or one still stamped with a
//!   superseded model. Such a row carries no distance, so it carries no
//!   semantic term and no activation score; it keeps the order the database
//!   ranked it in and follows the measured rows. See [`rank_page`].
//!
//! ## The cost this accepts, stated rather than left to be found
//!
//! On a store whose rows are embedded, the full-text arm no longer decides any
//! line of a full page: it fills the page only where the vector arm returned
//! fewer rows than were asked for. A query whose whole signal is lexical - an
//! identifier, a serial number, a quoted phrase an embedding represents poorly -
//! therefore loses the ranking help reciprocal-rank fusion gave it.
//!
//! What would give it back is the activation score's own full-text-rank term,
//! which `activation`'s documentation lists as awaiting an input because a
//! recall lookup uses one mode at a time and so supplies no rank. This path is
//! the first that supplies both, and #1239 is where that term is tracked. A
//! rank-shaped tiebreak bolted on here instead would reintroduce exactly what
//! this change removes.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::activation::{ActivationWeights, NO_SITUATION, activation};
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
/// `records` may be empty: a use log that could not be read costs the order and
/// never the page, exactly as it does on the recall path.
pub(crate) fn rank_page(
    candidates: Vec<SearchCandidate>,
    dispersion: RecallDispersion,
    records: &HashMap<String, KnowledgeUseRecord>,
    now: DateTime<Utc>,
    limit: usize,
) -> Vec<KnowledgeEntry> {
    let weights = ActivationWeights::default();
    let mut measured: Vec<(f64, KnowledgeEntry)> = Vec::new();
    let mut unmeasured: Vec<KnowledgeEntry> = Vec::new();

    for candidate in candidates {
        let Some(distance) = candidate.distance else {
            unmeasured.push(candidate.entry);
            continue;
        };
        let score = activation(
            dispersion.deviations_below_median(distance),
            records.get(&candidate.entry.id),
            NO_SITUATION,
            SalienceReading::read(&SalienceSource::of(&candidate.entry)).share(),
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

/// What [`crate::PgKnowledgeBaseStore::search`]'s hybrid arm reads.
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
/// `lexical` excludes every row `d` reached, by id, so the two lists never hold
/// the same row and a row the vector arm can compare is never ranked as though
/// it could not be. It is cut to `$5` - the caller's own page size - because
/// nothing reorders it: at most a whole page of such rows can ever show.
/// `measured` is cut to `$3`, which over-fetches, because activation reorders
/// it and a row it lifts has to be in the set to be lifted.
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
/// Held as its own string so the projection can be asserted on without a
/// database - see `the_hybrid_scan_measures_before_it_reads_any_entry`.
pub(crate) const HYBRID_SEARCH_SQL: &str = "\
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
                count(*) AS rows_read
         FROM d
     ),
     s AS (
         SELECT m.median,
                m.rows_read,
                (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(d.distance - m.median))
                 FROM d) AS deviation
         FROM m
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
                NULL::float8 AS distance,
                row_number() OVER (ORDER BY ts_rank_cd(kb.tsv, query) DESC,
                                            kb.updated_at DESC, kb.id DESC) AS seat
         FROM knowledge_base kb, plainto_tsquery('english', $4) query
         WHERE kb.user_id = $6
           AND kb.deleted_at IS NULL
           AND ($2::text[] IS NULL OR kb.tags && $2)
           AND ($7::text[] IS NULL OR NOT (kb.tags && $7))
           AND kb.tsv @@ query
           AND NOT EXISTS (SELECT 1 FROM d WHERE d.id = kb.id)
         ORDER BY ts_rank_cd(kb.tsv, query) DESC, kb.updated_at DESC, kb.id DESC
         LIMIT $5
     ),
     admitted AS (
         SELECT id, distance, 0 AS arm, seat FROM measured
         UNION ALL
         SELECT id, distance, 1 AS arm, seat FROM lexical
     )
     SELECT kb.id, kb.content, kb.tags, kb.metadata, kb.created_at, kb.updated_at, kb.summary,
            a.distance, s.median, s.rows_read, s.deviation
     FROM admitted a
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
        }
    }

    fn lexical(id: &str) -> SearchCandidate {
        SearchCandidate {
            entry: an_entry(id),
            distance: None,
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

    /// Acceptance (#1167): the page is ordered by the activation score, and not
    /// by any fusion of the two arms' ranks.
    ///
    /// The distinguishing case, because it is one no rank fusion can express: a
    /// candidate the vector arm ranked *second* takes the top line on what the
    /// use log knows about it. A fused rank has discarded both the distance and
    /// the log by the time it is a position, so under fusion the nearer row
    /// leads whatever its history.
    #[test]
    fn the_search_page_is_ordered_by_the_activation_score_and_not_by_a_fused_rank() {
        let records = log(vec![used("used", 12, 600)]);

        let page = rank_page(
            vec![measured("nearest", 0.50), measured("used", 0.52)],
            a_store(),
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
            &records,
            now(),
            10,
        );
        let loose = rank_page(
            candidates,
            RecallDispersion::assumed(0.80, 3.0),
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
            &HashMap::new(),
            now(),
            10,
        );

        assert_eq!(ids(&page), vec!["near", "far"]);
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
        let lexical = HYBRID_SEARCH_SQL
            .split("lexical AS (")
            .nth(1)
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
