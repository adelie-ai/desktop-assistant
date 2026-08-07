//! The knowledge use log (#698): what was offered, what was opened, and what
//! was marked.
//!
//! Every rule the log rests on lives in a `WHERE` clause or a conflict target,
//! so none of them can be proved without a real database. Each acceptance
//! criterion from the issue is one named test below:
//!
//! - `an_entry_can_be_marked_useful_and_the_mark_is_user_scoped`
//! - `offering_opening_and_marking_are_recorded_separately`
//! - `a_fetch_records_an_open_only_for_an_entry_offered_in_the_same_turn`
//! - `a_negative_mark_is_recordable_and_lowers_the_score`
//! - `the_recent_use_window_does_not_grow_without_limit`
//! - `a_cross_tenant_read_of_a_use_record_returns_nothing`
//!
//! Plus guards beyond that list:
//!
//! - `an_offer_standing_in_two_conversations_can_be_taken_up_in_either` - the
//!   standing offer is keyed on the conversation, so one conversation's offer
//!   cannot overwrite another's and silently drop the open made against it.
//! - `a_conversation_cannot_accumulate_standing_offers_without_limit`
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_use_log
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use chrono::Utc;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::knowledge_use::{
    KnowledgeUseRecord, MARK_REASON_MAX_CHARS, MarkPolarity, MarkSource, RECENT_USE_WINDOW,
    UseScoreWeights,
};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::knowledge_use::{
    KnowledgeUseLog, MAX_STANDING_OFFERS, MarkRequest, OfferScope,
};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgKnowledgeUseLog, UserId, with_user_id};
use sqlx::PgPool;

const ALICE: &str = "use-log-alice";
const BOB: &str = "use-log-bob";
const CONV: &str = "conv-1";

/// Boot a fixture in its own schema with migrations applied. `None` when
/// `TEST_DATABASE_URL` is unset, which is how each test pass-skips.
async fn fixture() -> Option<support::DbFixture> {
    support::DbFixture::try_new("uselog698").await
}

/// Write one entry as `user`, so the log has something it is allowed to record
/// against.
async fn write_as(pool: &PgPool, user: &str, id: &str) {
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(id, "a fact worth keeping", vec![]))
            .await
            .unwrap_or_else(|e| panic!("write {id}: {e}"));
    })
    .await;
}

/// The log's record for `id` as `user`, or `None` when it holds none.
async fn record_of(log: &PgKnowledgeUseLog, user: &str, id: &str) -> Option<KnowledgeUseRecord> {
    with_user_id(UserId::new(user), async {
        log.records(vec![id.to_string()])
            .await
            .expect("records read succeeds")
    })
    .await
    .into_iter()
    .next()
}

#[tokio::test]
async fn an_entry_can_be_marked_useful_and_the_mark_is_user_scoped() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-alice").await;
    write_as(&fx.pool, BOB, "kb-bob").await;

    let marked = with_user_id(UserId::new(ALICE), async {
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-alice".to_string()],
            polarity: MarkPolarity::Positive,
            source: MarkSource::Model,
            reason: Some("settled the question".to_string()),
        })
        .await
        .expect("mark succeeds")
    })
    .await;
    assert_eq!(marked, vec!["kb-alice".to_string()]);

    let record = record_of(&log, ALICE, "kb-alice").await.expect("a record");
    assert_eq!(record.marked_count, 1);
    let mark = record.standing_mark().expect("a standing mark");
    assert_eq!(mark.polarity, MarkPolarity::Positive);
    assert_eq!(mark.source, MarkSource::Model);
    assert_eq!(mark.reason.as_deref(), Some("settled the question"));

    // `knowledge_base.id` is a global primary key, so Bob can name Alice's id
    // exactly. Naming it must mark nothing.
    let stolen = with_user_id(UserId::new(BOB), async {
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-alice".to_string()],
            polarity: MarkPolarity::Negative,
            source: MarkSource::Model,
            reason: None,
        })
        .await
        .expect("a mark on another tenant's id succeeds and marks nothing")
    })
    .await;
    assert!(stolen.is_empty(), "bob must not mark alice's entry");
    assert_eq!(
        record_of(&log, ALICE, "kb-alice")
            .await
            .expect("a record")
            .standing_mark()
            .expect("a standing mark")
            .polarity,
        MarkPolarity::Positive,
        "alice's mark must survive bob naming her id"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn offering_opening_and_marking_are_recorded_separately() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        // Offered three times, taken up once. The ratio is the number the
        // ranking work reads, so the two counters must move independently.
        for _ in 0..3 {
            log.record_offered(OfferScope::recall(CONV), vec!["kb-1".to_string()])
                .await
                .expect("offer recorded");
        }
        log.record_opened(CONV.to_string(), vec!["kb-1".to_string()])
            .await
            .expect("open recorded");
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-1".to_string()],
            polarity: MarkPolarity::Positive,
            source: MarkSource::Model,
            reason: None,
        })
        .await
        .expect("mark recorded");
    })
    .await;

    let record = record_of(&log, ALICE, "kb-1").await.expect("a record");
    assert_eq!(record.offered_count, 3);
    assert_eq!(record.opened_count, 1);
    assert_eq!(record.marked_count, 1);
    assert_eq!(record.take_up_rate(), Some(1.0 / 3.0));
    assert!(record.last_offered_at.is_some());
    // An open and a mark are both uses; an offer is not.
    assert_eq!(record.recent_uses.len(), 2);

    fx.cleanup().await;
}

#[tokio::test]
async fn a_fetch_records_an_open_only_for_an_entry_offered_in_the_same_turn() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-offered").await;
    write_as(&fx.pool, ALICE, "kb-unrelated").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-offered".to_string()])
            .await
            .expect("offer recorded");

        // Both ids are fetched. Only the offered one is an open.
        let opened = log
            .record_opened(
                CONV.to_string(),
                vec!["kb-offered".to_string(), "kb-unrelated".to_string()],
            )
            .await
            .expect("open recorded");
        assert_eq!(opened, 1, "a read nothing offered is not an open");

        // The same fetch again, inside the same turn. The offer is already
        // taken up, so the second read adds nothing.
        let again = log
            .record_opened(CONV.to_string(), vec!["kb-offered".to_string()])
            .await
            .expect("repeat read succeeds");
        assert_eq!(again, 0, "a repeated fetch is one open, not two");
    })
    .await;

    assert_eq!(
        record_of(&log, ALICE, "kb-offered")
            .await
            .expect("a record")
            .opened_count,
        1
    );
    assert!(
        record_of(&log, ALICE, "kb-unrelated").await.is_none(),
        "an entry nothing offered has no record at all"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn an_offer_in_one_conversation_is_not_taken_up_in_another() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall("conv-a"), vec!["kb-1".to_string()])
            .await
            .expect("offer recorded");
        let elsewhere = log
            .record_opened("conv-b".to_string(), vec!["kb-1".to_string()])
            .await
            .expect("read succeeds");
        assert_eq!(
            elsewhere, 0,
            "an offer belongs to the conversation it was made in"
        );

        let here = log
            .record_opened("conv-a".to_string(), vec!["kb-1".to_string()])
            .await
            .expect("read succeeds");
        assert_eq!(here, 1);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn a_new_recall_block_replaces_the_previous_turns_standing_offers() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-last-turn").await;
    write_as(&fx.pool, ALICE, "kb-this-turn").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-last-turn".to_string()])
            .await
            .expect("offer recorded");
        // The next turn's block. It is this turn's whole offer set, so what
        // stood before it belongs to a turn that is over.
        log.record_offered(OfferScope::recall(CONV), vec!["kb-this-turn".to_string()])
            .await
            .expect("offer recorded");

        assert_eq!(
            log.record_opened(CONV.to_string(), vec!["kb-last-turn".to_string()])
                .await
                .expect("read succeeds"),
            0,
            "last turn's offer must not still be standing"
        );
        assert_eq!(
            log.record_opened(CONV.to_string(), vec!["kb-this-turn".to_string()])
                .await
                .expect("read succeeds"),
            1
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn a_turn_that_offered_nothing_still_ends_the_previous_turns_offers() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-1".to_string()])
            .await
            .expect("offer recorded");
        // The next prompt had nothing near it, so the block showed no entry.
        // That is still this turn's whole offer set, and it is what ends the
        // previous turn's.
        log.record_offered(OfferScope::recall(CONV), vec![])
            .await
            .expect("empty offer recorded");

        assert_eq!(
            log.record_opened(CONV.to_string(), vec!["kb-1".to_string()])
                .await
                .expect("read succeeds"),
            0,
            "an offer must not outlive the turn that made it"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn a_search_inside_a_turn_adds_to_what_the_block_already_offered() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-recalled").await;
    write_as(&fx.pool, ALICE, "kb-searched").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-recalled".to_string()])
            .await
            .expect("offer recorded");
        // A search runs inside the turn the block opened, so it must not wipe
        // what the block offered.
        log.record_offered(OfferScope::search(CONV), vec!["kb-searched".to_string()])
            .await
            .expect("offer recorded");

        let opened = log
            .record_opened(
                CONV.to_string(),
                vec!["kb-recalled".to_string(), "kb-searched".to_string()],
            )
            .await
            .expect("read succeeds");
        assert_eq!(opened, 2, "both offers must still stand");
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn an_offer_standing_in_two_conversations_can_be_taken_up_in_either() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-broad").await;

    with_user_id(UserId::new(ALICE), async {
        // An entry broad enough to rank near two different prompts is offered
        // in both conversations. Neither offer may displace the other: the
        // entries that surface in two places at once are exactly the ones a
        // single-conversation column would undercount.
        log.record_offered(OfferScope::recall("conv-a"), vec!["kb-broad".to_string()])
            .await
            .expect("offer recorded");
        log.record_offered(OfferScope::recall("conv-b"), vec!["kb-broad".to_string()])
            .await
            .expect("offer recorded");

        assert_eq!(
            log.record_opened("conv-a".to_string(), vec!["kb-broad".to_string()])
                .await
                .expect("read succeeds"),
            1,
            "the first conversation's offer must still stand"
        );
        assert_eq!(
            log.record_opened("conv-b".to_string(), vec!["kb-broad".to_string()])
                .await
                .expect("read succeeds"),
            1,
            "and so must the second's"
        );
    })
    .await;

    let record = record_of(&log, ALICE, "kb-broad").await.expect("a record");
    assert_eq!(record.offered_count, 2);
    assert_eq!(record.opened_count, 2);

    fx.cleanup().await;
}

#[tokio::test]
async fn a_conversation_cannot_accumulate_standing_offers_without_limit() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());

    // A deployment that renders no [Recall] block never clears at a turn
    // boundary, so only the cap bounds a long conversation's standing offers.
    let over = MAX_STANDING_OFFERS + 20;
    let ids: Vec<String> = (0..over).map(|i| format!("kb-{i:04}")).collect();
    for id in &ids {
        write_as(&fx.pool, ALICE, id).await;
    }
    with_user_id(UserId::new(ALICE), async {
        for id in &ids {
            log.record_offered(OfferScope::search(CONV), vec![id.clone()])
                .await
                .expect("offer recorded");
        }
    })
    .await;

    let standing: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM knowledge_offers WHERE user_id = $1 AND conversation_id = $2",
    )
    .bind(ALICE)
    .bind(CONV)
    .fetch_one(&fx.pool)
    .await
    .expect("count standing offers");
    assert_eq!(standing, MAX_STANDING_OFFERS as i64);

    // The cap keeps the newest, which is where a take-up would come from.
    with_user_id(UserId::new(ALICE), async {
        assert_eq!(
            log.record_opened(CONV.to_string(), vec![ids[over - 1].clone()])
                .await
                .expect("read succeeds"),
            1,
            "the newest offer must survive the trim"
        );
        assert_eq!(
            log.record_opened(CONV.to_string(), vec![ids[0].clone()])
                .await
                .expect("read succeeds"),
            0,
            "the oldest was pushed out"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn a_negative_mark_is_recordable_and_lowers_the_score() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-right").await;
    write_as(&fx.pool, ALICE, "kb-wrong").await;

    with_user_id(UserId::new(ALICE), async {
        for id in ["kb-right", "kb-wrong"] {
            log.record_offered(OfferScope::recall(CONV), vec![id.to_string()])
                .await
                .expect("offer recorded");
            log.record_opened(CONV.to_string(), vec![id.to_string()])
                .await
                .expect("open recorded");
        }
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-wrong".to_string()],
            polarity: MarkPolarity::Negative,
            source: MarkSource::Model,
            reason: Some("named a host that no longer exists".to_string()),
        })
        .await
        .expect("negative mark recorded");
    })
    .await;

    let wrong = record_of(&log, ALICE, "kb-wrong").await.expect("a record");
    let right = record_of(&log, ALICE, "kb-right").await.expect("a record");
    let mark = wrong.standing_mark().expect("a standing mark");
    assert_eq!(mark.polarity, MarkPolarity::Negative);
    assert_eq!(
        mark.reason.as_deref(),
        Some("named a host that no longer exists")
    );

    let now = Utc::now();
    let weights = UseScoreWeights::default();
    assert!(
        wrong.usefulness(now, &weights) < right.usefulness(now, &weights),
        "an entry marked wrong must score below the same entry unmarked"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_long_reason_is_cut_rather_than_refused() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    // The reason comes from a language model and nothing before storage bounds
    // it. Cutting costs the tail; refusing would cost the mark, which is the
    // highest-quality signal the log holds.
    let long: String = "e".repeat(MARK_REASON_MAX_CHARS * 3);
    with_user_id(UserId::new(ALICE), async {
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-1".to_string()],
            polarity: MarkPolarity::Negative,
            source: MarkSource::Model,
            reason: Some(long),
        })
        .await
        .expect("a long reason does not refuse the mark")
    })
    .await;

    let record = record_of(&log, ALICE, "kb-1").await.expect("a record");
    let stored = record.marks[0].reason.as_deref().expect("a reason");
    assert_eq!(stored.chars().count(), MARK_REASON_MAX_CHARS);

    fx.cleanup().await;
}

#[tokio::test]
async fn a_second_mark_from_one_source_replaces_the_first() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        for polarity in [MarkPolarity::Negative, MarkPolarity::Positive] {
            log.record_mark(MarkRequest {
                entry_ids: vec!["kb-1".to_string()],
                polarity,
                source: MarkSource::Model,
                reason: None,
            })
            .await
            .expect("mark recorded");
        }
    })
    .await;

    let record = record_of(&log, ALICE, "kb-1").await.expect("a record");
    assert_eq!(record.marks.len(), 1, "one standing mark per source");
    assert_eq!(record.marks[0].polarity, MarkPolarity::Positive);
    assert_eq!(
        record.marked_count, 2,
        "both marks were acts, and both count"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn the_recent_use_window_does_not_grow_without_limit() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    let uses = RECENT_USE_WINDOW * 3;
    with_user_id(UserId::new(ALICE), async {
        for _ in 0..uses {
            log.record_offered(OfferScope::recall(CONV), vec!["kb-1".to_string()])
                .await
                .expect("offer recorded");
            log.record_opened(CONV.to_string(), vec!["kb-1".to_string()])
                .await
                .expect("open recorded");
        }
    })
    .await;

    let record = record_of(&log, ALICE, "kb-1").await.expect("a record");
    assert_eq!(record.opened_count, uses as u64);
    assert_eq!(
        record.recent_uses.len(),
        RECENT_USE_WINDOW,
        "the exact window is capped however many times the entry is used"
    );
    // Newest first: the window keeps the recent end, which is the end a
    // spacing term reads.
    let mut sorted = record.recent_uses.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(record.recent_uses, sorted);

    fx.cleanup().await;
}

#[tokio::test]
async fn a_cross_tenant_read_of_a_use_record_returns_nothing() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-alice").await;
    write_as(&fx.pool, BOB, "kb-bob").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-alice".to_string()])
            .await
            .expect("offer recorded");
        log.record_opened(CONV.to_string(), vec!["kb-alice".to_string()])
            .await
            .expect("open recorded");
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-alice".to_string()],
            polarity: MarkPolarity::Positive,
            source: MarkSource::Model,
            reason: Some("alice's own judgement".to_string()),
        })
        .await
        .expect("mark recorded");
    })
    .await;

    // Bob names Alice's id exactly, in every read and write the log has.
    let bobs_view = with_user_id(UserId::new(BOB), async {
        log.records(vec!["kb-alice".to_string(), "kb-bob".to_string()])
            .await
            .expect("records read succeeds")
    })
    .await;
    assert!(
        bobs_view.is_empty(),
        "bob must see no record for alice's entry, and has none of his own"
    );

    let bobs_open = with_user_id(UserId::new(BOB), async {
        log.record_opened(CONV.to_string(), vec!["kb-alice".to_string()])
            .await
            .expect("read succeeds")
    })
    .await;
    assert_eq!(bobs_open, 0, "bob must not take up alice's standing offer");

    let bobs_offer = with_user_id(UserId::new(BOB), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-alice".to_string()])
            .await
            .expect("offer succeeds")
    })
    .await;
    assert_eq!(
        bobs_offer, 0,
        "bob must not record an offer of alice's entry"
    );

    // Alice's own record is untouched by everything Bob did.
    let alices = record_of(&log, ALICE, "kb-alice").await.expect("a record");
    assert_eq!(alices.offered_count, 1);
    assert_eq!(alices.opened_count, 1);
    assert_eq!(alices.marks.len(), 1);

    fx.cleanup().await;
}

#[tokio::test]
async fn the_log_records_nothing_for_an_entry_that_cannot_be_read() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-live").await;
    sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind("kb-retired")
        .execute(&fx.pool)
        .await
        .expect("no-op update");
    write_as(&fx.pool, ALICE, "kb-retired").await;
    sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind("kb-retired")
        .execute(&fx.pool)
        .await
        .expect("retire");

    let written = with_user_id(UserId::new(ALICE), async {
        log.record_offered(
            OfferScope::recall(CONV),
            vec![
                "kb-live".to_string(),
                "kb-retired".to_string(),
                "kb-never-existed".to_string(),
                // An id no stored id can equal. Sent to the database it raises
                // and takes the whole batch with it, so it must be dropped.
                "kb\u{0}broken".to_string(),
            ],
        )
        .await
        .expect("offer succeeds")
    })
    .await;
    assert_eq!(written, 1, "only the live, owned entry is recorded");

    fx.cleanup().await;
}

#[tokio::test]
async fn an_empty_batch_is_a_successful_no_op() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());

    with_user_id(UserId::new(ALICE), async {
        assert_eq!(
            log.record_offered(OfferScope::recall(CONV), vec![])
                .await
                .expect("empty offer succeeds"),
            0
        );
        assert_eq!(
            log.record_opened(CONV.to_string(), vec![])
                .await
                .expect("empty open succeeds"),
            0
        );
        assert!(
            log.record_mark(MarkRequest {
                entry_ids: vec![],
                polarity: MarkPolarity::Positive,
                source: MarkSource::Model,
                reason: None,
            })
            .await
            .expect("empty mark succeeds")
            .is_empty()
        );
        assert!(
            log.records(vec![])
                .await
                .expect("empty read succeeds")
                .is_empty()
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn a_persons_mark_and_the_models_are_held_apart() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-1".to_string()],
            polarity: MarkPolarity::Positive,
            source: MarkSource::Model,
            reason: None,
        })
        .await
        .expect("model mark recorded");
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-1".to_string()],
            polarity: MarkPolarity::Negative,
            source: MarkSource::Person,
            reason: Some("this has been wrong since the move".to_string()),
        })
        .await
        .expect("person mark recorded");
    })
    .await;

    let record = record_of(&log, ALICE, "kb-1").await.expect("a record");
    assert_eq!(record.marks.len(), 2, "one standing mark per source");
    assert_eq!(
        record.standing_mark().expect("a standing mark").source,
        MarkSource::Person,
        "a person's mark outranks the model's"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn reaping_an_entry_frees_its_use_records() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-1".to_string()])
            .await
            .expect("offer recorded");
        log.record_mark(MarkRequest {
            entry_ids: vec!["kb-1".to_string()],
            polarity: MarkPolarity::Positive,
            source: MarkSource::Model,
            reason: None,
        })
        .await
        .expect("mark recorded");
    })
    .await;

    // The hard reap that empties the trash. The log must not hold rows for an
    // entry that no longer exists, and must not stand in the reap's way.
    sqlx::query("DELETE FROM knowledge_base WHERE id = $1")
        .bind("kb-1")
        .execute(&fx.pool)
        .await
        .expect("hard delete");

    assert!(record_of(&log, ALICE, "kb-1").await.is_none());
    for table in ["knowledge_use_marks", "knowledge_offers"] {
        let left: i64 =
            sqlx::query_scalar(sqlx::AssertSqlSafe(format!("SELECT count(*) FROM {table}")))
                .fetch_one(&fx.pool)
                .await
                .unwrap_or_else(|e| panic!("count {table}: {e}"));
        assert_eq!(left, 0, "{table} must not outlive the entry it describes");
    }

    fx.cleanup().await;
}
