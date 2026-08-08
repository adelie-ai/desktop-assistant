//! What the daily pass should look at first (#1127).
//!
//! Consolidation reads the whole active store every night and asks a strong
//! model to recompute it. Nothing tells it where to start, so it starts at the
//! first tag in alphabetical order and works down - which means the material it
//! examines while the pass is healthy is decided by a string sort, and the
//! material it never reaches when a pass is cancelled or a slice fails is
//! decided by the same sort.
//!
//! This module answers "what is worth re-examining today" on the same scale the
//! retrieval score uses, so the two cannot drift.
//!
//! ## The score
//!
//! ```text
//! P_i = retrieved + contradicted + salient
//! ```
//!
//! - **`retrieved`** is how much the entry has actually been reached for,
//!   through [`KnowledgeUseRecord::retrieval_sum`] read on the reinforcement
//!   scale. **Retrieved, not written**, and that is the whole point of the term.
//!   Reconsolidation makes a memory editable at the moment it is recalled, which
//!   is where a contradiction surfaces; a fact written yesterday and never
//!   reached for has told nobody anything yet. Write activity cannot identify
//!   the first case, and the use log (#698) already carries it.
//! - **`contradicted`** is how much standing evidence says the entry is wrong,
//!   through [`KnowledgeUseRecord::contradiction_sum`] on the same scale. It
//!   **adds** here, where the retrieval score subtracts it. A fact that was
//!   retrieved and then contradicted is the highest-value thing a consolidation
//!   pass can examine, and the ranking that pushes it out of a `[Recall]` block
//!   is the ranking that should pull it to the front of a review.
//! - **`salient`** is how much of the salience information this build can detect
//!   the entry carries, through [`ActivationWeights::salience`].
//!   [`crate::domain::salience`] states the signals and why the bound is a scale.
//!
//! ## One currency, and it is not a second set of weights
//!
//! Every term is read through [`ActivationWeights`], the same struct the
//! `[Recall]` block scores with. Nothing here introduces a coefficient: the
//! retrieval and contradiction terms are the reinforcement function applied to
//! two halves of the use log, and the salience term is the reference-use lift
//! this project already spends every cheap signal against. A deployment that
//! fits its own `use_lift` moves the retrieval block and the daily pass together.
//!
//! ## It orders, it does not select
//!
//! The pass still examines every active entry. What this decides is the order,
//! and therefore two things: which entries share a slice, and - because a pass
//! stops between slices when it is cancelled and continues past a slice that
//! failed - which entries a pass that did not finish got to. Capping a day's
//! material is #894's to decide, and a cap laid over an arbitrary order would
//! silently stop examining whatever sorted last.

use chrono::{DateTime, Utc};

use crate::domain::activation::ActivationWeights;
use crate::domain::knowledge_use::KnowledgeUseRecord;

/// An entry no salience detector says anything about, which is what a caller
/// passes where it has no text to read.
pub const NO_SALIENCE_SHARE: f64 = 0.0;

/// What re-examining one entry is worth today.
///
/// `record` is what the use log knows about it, and `None` - an entry nothing
/// has ever offered, opened or marked - contributes exactly zero, so a store
/// with no use history is ordered by salience alone and, failing that, by
/// whatever order it arrived in. `salience_share` is
/// [`SalienceReading::share`](crate::domain::salience::SalienceReading::share),
/// and [`NO_SALIENCE_SHARE`] is the answer where there is no text to read.
///
/// Never negative: a contradiction raises this score rather than sinking it,
/// which is the one place this function deliberately disagrees with
/// [`crate::domain::activation`].
pub fn replay_priority(
    record: Option<&KnowledgeUseRecord>,
    salience_share: f64,
    now: DateTime<Utc>,
    weights: &ActivationWeights,
) -> f64 {
    let retrieved = record.map_or(0.0, |record| {
        weights.reinforcement(record.retrieval_sum(now, &weights.use_score))
    });
    let contradicted = record.map_or(0.0, |record| {
        weights.reinforcement(record.contradiction_sum(now, &weights.use_score))
    });
    retrieved + contradicted + weights.salience(salience_share)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::knowledge_use::{KnowledgeMark, MarkPolarity, MarkSource};
    use chrono::TimeDelta;

    const DAY: i64 = 24 * 3600;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-07T12:00:00Z")
            .expect("a fixed clock parses")
            .with_timezone(&Utc)
    }

    fn at(now: DateTime<Utc>, seconds_ago: i64) -> DateTime<Utc> {
        now - TimeDelta::seconds(seconds_ago)
    }

    /// An entry retrieved at each of `ages`, and never marked.
    fn retrieved(now: DateTime<Utc>, ages: &[i64]) -> KnowledgeUseRecord {
        KnowledgeUseRecord {
            entry_id: "kb-1".to_string(),
            offered_count: ages.len() as u64,
            opened_count: ages.len() as u64,
            marked_count: 0,
            first_seen_at: at(now, ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(at(now, ages.iter().copied().min().unwrap_or(1))),
            recent_uses: ages.iter().map(|a| at(now, *a)).collect(),
            marks: Vec::new(),
        }
    }

    /// The same record, plus a standing negative mark set `age` seconds ago. As
    /// the writer stores it: a mark is a use whichever way it points, so the
    /// counter moves too.
    fn contradicted(
        mut record: KnowledgeUseRecord,
        now: DateTime<Utc>,
        age: i64,
    ) -> KnowledgeUseRecord {
        record.marked_count += 1;
        record.marks.push(KnowledgeMark {
            source: MarkSource::Model,
            polarity: MarkPolarity::Negative,
            reason: Some("the fact it states was withdrawn".to_string()),
            marked_at: at(now, age),
        });
        record
    }

    /// Acceptance (#1127): the daily pass prioritises what was **retrieved**
    /// recently, which write activity cannot identify.
    ///
    /// The entry nothing has ever reached for is the one written yesterday, and
    /// it still comes second - because being written is not evidence that
    /// anybody wanted it.
    #[test]
    fn an_entry_that_was_retrieved_outranks_one_that_was_only_written() {
        let now = now();
        let weights = ActivationWeights::default();

        let reached_for = replay_priority(
            Some(&retrieved(now, &[3_600])),
            NO_SALIENCE_SHARE,
            now,
            &weights,
        );
        let only_written = replay_priority(None, NO_SALIENCE_SHARE, now, &weights);

        assert!(
            reached_for > only_written,
            "an entry opened an hour ago scored {reached_for} and one nothing has reached for \
             scored {only_written}"
        );
    }

    /// Acceptance (#1127): a fact that was retrieved and then contradicted is
    /// prioritised over one that was merely retrieved.
    ///
    /// The two histories are identical apart from the mark, so the contradiction
    /// is the only thing that separates them.
    #[test]
    fn a_retrieved_and_contradicted_entry_outranks_one_that_was_merely_retrieved() {
        let now = now();
        let weights = ActivationWeights::default();
        let history = retrieved(now, &[600, 6_000]);

        let merely = replay_priority(Some(&history), NO_SALIENCE_SHARE, now, &weights);
        let wrong = contradicted(history, now, 600);
        let and_wrong = replay_priority(Some(&wrong), NO_SALIENCE_SHARE, now, &weights);

        assert!(
            and_wrong > merely,
            "the contradicted entry scored {and_wrong} and the merely retrieved one scored \
             {merely}"
        );
    }

    /// The one place this score deliberately disagrees with the retrieval score:
    /// a contradiction **raises** replay priority where it lowers activation.
    ///
    /// Stated as its own test because the two signs are easy to unify by
    /// accident, and unifying them would hide the most valuable thing a
    /// consolidation pass can examine behind every entry nobody has ever opened.
    #[test]
    fn a_contradiction_raises_replay_priority_where_it_lowers_the_retrieval_score() {
        use crate::domain::activation::{NO_SITUATION, activation};

        let now = now();
        let weights = ActivationWeights::default();
        let history = retrieved(now, &[600]);
        let wrong = contradicted(history.clone(), now, 600);

        assert!(
            replay_priority(Some(&wrong), NO_SALIENCE_SHARE, now, &weights)
                > replay_priority(Some(&history), NO_SALIENCE_SHARE, now, &weights),
            "a contradiction must pull an entry toward the front of a review"
        );
        assert!(
            activation(7.0, Some(&wrong), NO_SITUATION, 0.0, now, &weights)
                < activation(7.0, Some(&history), NO_SITUATION, 0.0, now, &weights),
            "and must still push it out of a [Recall] block"
        );
    }

    /// Acceptance (#1127): a salient item is prioritised ahead of a non-salient
    /// item of the same age.
    ///
    /// Same age and the same history - here, no history at all, which is the
    /// state most of a store is in - so salience is the only thing that
    /// separates them.
    #[test]
    fn a_salient_entry_outranks_a_non_salient_entry_of_the_same_age() {
        let now = now();
        let weights = ActivationWeights::default();

        let salient = replay_priority(None, 1.0, now, &weights);
        let plain = replay_priority(None, NO_SALIENCE_SHARE, now, &weights);
        assert!(salient > plain, "{salient} against {plain}");

        // And with equal histories as well, so the claim is not confined to a
        // cold store.
        let history = retrieved(now, &[3_600, 36_000]);
        assert!(
            replay_priority(Some(&history), 1.0, now, &weights)
                > replay_priority(Some(&history), NO_SALIENCE_SHARE, now, &weights)
        );
    }

    /// A contradiction nobody has acted on for a year is less urgent than one
    /// from this morning, and it decays for that reason rather than because it
    /// has stopped being true.
    #[test]
    fn an_old_contradiction_is_worth_less_than_a_fresh_one() {
        let now = now();
        let weights = ActivationWeights::default();
        let history = retrieved(now, &[600]);

        let fresh = replay_priority(
            Some(&contradicted(history.clone(), now, 600)),
            NO_SALIENCE_SHARE,
            now,
            &weights,
        );
        let stale = replay_priority(
            Some(&contradicted(history.clone(), now, 365 * DAY)),
            NO_SALIENCE_SHARE,
            now,
            &weights,
        );
        let none = replay_priority(Some(&history), NO_SALIENCE_SHARE, now, &weights);

        assert!(stale < fresh, "{stale} against {fresh}");
        assert!(
            stale > none,
            "a year-old contradiction is still a contradiction: {stale} against {none}"
        );
    }

    /// An entry nothing has touched and no detector reads has no priority of its
    /// own, so a store with neither is ordered exactly as it was.
    #[test]
    fn an_entry_nothing_has_touched_has_no_replay_priority_of_its_own() {
        let now = now();
        let weights = ActivationWeights::default();
        assert_eq!(replay_priority(None, NO_SALIENCE_SHARE, now, &weights), 0.0);
        assert_eq!(
            replay_priority(
                Some(&KnowledgeUseRecord::unseen("kb-1", now)),
                NO_SALIENCE_SHARE,
                now,
                &weights
            ),
            0.0
        );
    }

    /// Never negative, over every state the log can be in. A score that could go
    /// below zero would sort a contradicted entry behind one nobody has opened.
    #[test]
    fn replay_priority_is_never_negative() {
        let now = now();
        let weights = ActivationWeights::default();
        let mut only_a_negative_mark = KnowledgeUseRecord::unseen("kb-1", now);
        only_a_negative_mark.marked_count = 1;
        only_a_negative_mark.marks = vec![KnowledgeMark {
            source: MarkSource::Person,
            polarity: MarkPolarity::Negative,
            reason: None,
            marked_at: at(now, 60),
        }];

        for share in [0.0, 0.5, 1.0] {
            for record in [
                None,
                Some(&only_a_negative_mark),
                Some(&retrieved(now, &[60])),
            ] {
                let scored = replay_priority(record, share, now, &weights);
                assert!(scored >= 0.0, "scored {scored}");
                assert!(scored.is_finite());
            }
        }
    }
}
