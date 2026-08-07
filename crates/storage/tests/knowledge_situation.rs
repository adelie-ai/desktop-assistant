//! The situation as a retrieval cue (#1125, absorbing #238): what an entry was
//! seen in, and what the present situation is worth over this store.
//!
//! Every rule here lives in a conflict target, a `WHERE` clause or an aggregate,
//! so none of them can be proved without a real database. Each acceptance
//! criterion the storage layer owns is one named test below:
//!
//! - `a_situation_record_is_written_with_every_observation_and_every_field_is_optional`
//! - `an_entry_accumulates_a_usage_context_on_reuse_without_rewriting_the_entry`
//! - `a_situation_the_entry_already_holds_records_no_second_value`
//! - `the_cue_counts_the_whole_store_and_not_one_lookups_candidates`
//! - `a_value_the_whole_store_shares_is_measured_as_worth_nothing`
//! - `a_cross_tenant_read_of_a_situation_record_returns_nothing`
//! - `an_entry_cannot_accumulate_situation_values_without_limit`
//! - `deleting_an_entry_takes_its_situation_rows_with_it`
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_situation
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::collections::BTreeMap;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::situation::{
    MAX_SITUATION_VALUES_PER_FIELD, SITUATION_MIN_POPULATION, Situation, SituationField,
    SituationRecord,
};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::knowledge_use::{KnowledgeUseLog, OfferScope};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgKnowledgeUseLog, UserId, with_user_id};
use sqlx::PgPool;

const ALICE: &str = "situation-alice";
const BOB: &str = "situation-bob";
const CONV: &str = "conv-1";

async fn fixture() -> Option<support::DbFixture> {
    support::DbFixture::try_new("situation1125").await
}

/// Write one entry as `user`, so the log has something to record against.
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

/// The situation record the log holds for `id` as `user`.
async fn situation_of(log: &PgKnowledgeUseLog, user: &str, id: &str) -> Option<SituationRecord> {
    with_user_id(UserId::new(user), async {
        log.situations(vec![id.to_string()])
            .await
            .expect("situations read succeeds")
    })
    .await
    .into_iter()
    .next()
    .map(|(_, record)| record)
}

/// A situation at `host`, on `weekday`.
fn at(host: &str, weekday: &str) -> Situation {
    Situation::new()
        .with(SituationField::Host, host)
        .with(SituationField::Weekday, weekday)
}

/// Acceptance (#1125): a situation record is written with every observation,
/// with every field optional.
///
/// Three writes: one that states both fields, one that states a single field,
/// and one that states none. Each is accepted; each records exactly what it
/// stated.
#[tokio::test]
async fn a_situation_record_is_written_with_every_observation_and_every_field_is_optional() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    for id in ["kb-both", "kb-one", "kb-none"] {
        write_as(&fx.pool, ALICE, id).await;
    }

    with_user_id(UserId::new(ALICE), async {
        log.record_situation(vec!["kb-both".to_string()], at("workshop", "thursday"))
            .await
            .expect("a full situation is recorded");
        log.record_situation(
            vec!["kb-one".to_string()],
            Situation::new().with(SituationField::Host, "workshop"),
        )
        .await
        .expect("a partial situation is recorded");
        log.record_situation(vec!["kb-none".to_string()], Situation::new())
            .await
            .expect("an empty situation is accepted and records nothing");
    })
    .await;

    let both = situation_of(&log, ALICE, "kb-both")
        .await
        .expect("a record");
    assert!(both.holds(SituationField::Host, "workshop"));
    assert!(both.holds(SituationField::Weekday, "thursday"));

    let one = situation_of(&log, ALICE, "kb-one").await.expect("a record");
    assert!(one.holds(SituationField::Host, "workshop"));
    assert!(
        !one.knows(SituationField::Weekday),
        "a field the observation could not read must stay absent, not be filled with a guess"
    );

    assert_eq!(
        situation_of(&log, ALICE, "kb-none").await,
        None,
        "an observation with nothing connected writes no row"
    );

    fx.cleanup().await;
}

/// Acceptance (#1125, absorbing #238): an entry accumulates a usage-context on
/// reuse, and the entry itself is not rewritten.
///
/// The situation travels with the write that already decides which ids count as
/// opens, so an entry that was standing offered accumulates and one that was not
/// does not. `knowledge_base.updated_at` must not move: an entry that had to be
/// rewritten to learn where it is useful would restate its own content, bump its
/// timestamp, and put itself back in the embedding backfill queue.
#[tokio::test]
async fn an_entry_accumulates_a_usage_context_on_reuse_without_rewriting_the_entry() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    write_as(&fx.pool, ALICE, "kb-offered").await;
    write_as(&fx.pool, ALICE, "kb-unoffered").await;

    let before = with_user_id(UserId::new(ALICE), async {
        store
            .get("kb-offered")
            .await
            .expect("read succeeds")
            .expect("the entry exists")
            .updated_at
    })
    .await;

    with_user_id(UserId::new(ALICE), async {
        log.record_offered(OfferScope::recall(CONV), vec!["kb-offered".to_string()])
            .await
            .expect("offer recorded");
        log.record_opened(
            CONV.to_string(),
            vec!["kb-offered".to_string(), "kb-unoffered".to_string()],
            at("the-boat", "sunday"),
        )
        .await
        .expect("open recorded");
    })
    .await;

    let opened = situation_of(&log, ALICE, "kb-offered")
        .await
        .expect("a record");
    assert!(opened.holds(SituationField::Host, "the-boat"));
    assert!(opened.holds(SituationField::Weekday, "sunday"));

    assert_eq!(
        situation_of(&log, ALICE, "kb-unoffered").await,
        None,
        "a read nothing offered is not a reuse, so it accumulates nothing"
    );

    let after = with_user_id(UserId::new(ALICE), async {
        store
            .get("kb-offered")
            .await
            .expect("read succeeds")
            .expect("the entry exists")
            .updated_at
    })
    .await;
    assert_eq!(
        before, after,
        "accumulating a usage-context must not rewrite the entry"
    );

    fx.cleanup().await;
}

/// A second reuse in a situation the entry already holds adds no second value,
/// which is what closes the retrieve-record-retrieve loop after one step.
///
/// A new value in the same field is added beside the first, because an entry
/// that is useful in two places is useful in two places.
#[tokio::test]
async fn a_situation_the_entry_already_holds_records_no_second_value() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        for _ in 0..3 {
            log.record_situation(
                vec!["kb-1".to_string()],
                Situation::new().with(SituationField::Host, "workshop"),
            )
            .await
            .expect("recorded");
        }
        log.record_situation(
            vec!["kb-1".to_string()],
            Situation::new().with(SituationField::Host, "the-boat"),
        )
        .await
        .expect("recorded");
    })
    .await;

    let record = situation_of(&log, ALICE, "kb-1").await.expect("a record");
    let hosts: Vec<&str> = record
        .iter()
        .filter(|(field, _)| *field == SituationField::Host)
        .map(|(_, value)| value)
        .collect();
    assert_eq!(
        hosts,
        vec!["the-boat", "workshop"],
        "three recordings of one host are one value; a second host is a second value"
    );

    fx.cleanup().await;
}

/// The cue is measured over the whole store, never over one lookup's
/// candidates, and a store too small to measure produces no cue at all.
#[tokio::test]
async fn the_cue_counts_the_whole_store_and_not_one_lookups_candidates() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    let population = SITUATION_MIN_POPULATION as usize + 20;

    // A store below the floor answers with no cue.
    write_as(&fx.pool, ALICE, "kb-first").await;
    with_user_id(UserId::new(ALICE), async {
        log.record_situation(vec!["kb-first".to_string()], at("workshop", "thursday"))
            .await
            .expect("recorded");
        assert_eq!(
            log.situation_cue(at("workshop", "thursday"))
                .await
                .expect("cue read succeeds"),
            None,
            "a store of one entry cannot grade a cue"
        );
    })
    .await;

    // A quarter of the store shares the cue's host; the rest do not.
    for i in 0..population {
        let id = format!("kb-{i}");
        write_as(&fx.pool, ALICE, &id).await;
        let host = if i % 4 == 0 { "workshop" } else { "the-road" };
        with_user_id(UserId::new(ALICE), async {
            log.record_situation(vec![id.clone()], at(host, "thursday"))
                .await
                .expect("recorded");
        })
        .await;
    }

    let cue = with_user_id(UserId::new(ALICE), async {
        log.situation_cue(at("workshop", "thursday"))
            .await
            .expect("cue read succeeds")
    })
    .await
    .expect("a store this size can grade a cue");

    assert!(
        cue.information(SituationField::Host) > 0.0,
        "a host a quarter of the store carries must separate something"
    );
    // The weekday is on every entry, so it separates nobody.
    assert_eq!(cue.information(SituationField::Weekday), 0.0);

    // And the measurement is the store's, not the candidates': an entry that
    // carries the cue's host scores full coverage against it.
    assert_eq!(
        cue.coverage(&SituationRecord::from_pairs([
            (SituationField::Host, "workshop"),
            (SituationField::Weekday, "thursday"),
        ])),
        1.0
    );
    assert_eq!(
        cue.coverage(&SituationRecord::from_pairs([
            (SituationField::Host, "the-road"),
            (SituationField::Weekday, "thursday"),
        ])),
        0.0
    );

    fx.cleanup().await;
}

/// Acceptance (#1125): the term's size does not grow with the number of
/// situation fields a deployment has connected - and the reason it cannot is
/// measured here rather than assumed. A value the whole store shares is counted
/// as worth nothing, so connecting a field that never varies adds nothing.
#[tokio::test]
async fn a_value_the_whole_store_shares_is_measured_as_worth_nothing() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    let population = SITUATION_MIN_POPULATION as usize + 5;

    for i in 0..population {
        let id = format!("kb-{i}");
        write_as(&fx.pool, ALICE, &id).await;
        with_user_id(UserId::new(ALICE), async {
            log.record_situation(
                vec![id.clone()],
                Situation::new().with(SituationField::Host, "the-only-host"),
            )
            .await
            .expect("recorded");
        })
        .await;
    }

    let cue = with_user_id(UserId::new(ALICE), async {
        log.situation_cue(Situation::new().with(SituationField::Host, "the-only-host"))
            .await
            .expect("cue read succeeds")
    })
    .await
    .expect("a store this size can grade a cue");

    assert!(
        cue.is_empty(),
        "one host on every entry separates nobody, so it must lift nobody"
    );
    assert_eq!(
        cue.coverage(&SituationRecord::from_pairs([(
            SituationField::Host,
            "the-only-host"
        )])),
        0.0
    );

    fx.cleanup().await;
}

/// A situation record is personal data, and a read scoped to another user
/// returns nothing - including for an id that user can name exactly.
#[tokio::test]
async fn a_cross_tenant_read_of_a_situation_record_returns_nothing() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-alice").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_situation(vec!["kb-alice".to_string()], at("workshop", "thursday"))
            .await
            .expect("recorded");
    })
    .await;

    // `knowledge_base.id` is a global primary key, so Bob can name Alice's id.
    assert_eq!(
        situation_of(&log, BOB, "kb-alice").await,
        None,
        "bob must not read where alice works"
    );

    let written = with_user_id(UserId::new(BOB), async {
        log.record_situation(vec!["kb-alice".to_string()], at("elsewhere", "monday"))
            .await
            .expect("a write against another tenant's id succeeds and writes nothing")
    })
    .await;
    assert_eq!(written, 0, "bob must not write onto alice's entry");

    let alice = situation_of(&log, ALICE, "kb-alice")
        .await
        .expect("a record");
    assert!(!alice.holds(SituationField::Host, "elsewhere"));

    fx.cleanup().await;
}

/// One entry's record is bounded however many places it proves useful in. The
/// least recently seen value falls out first.
#[tokio::test]
async fn an_entry_cannot_accumulate_situation_values_without_limit() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    let over = MAX_SITUATION_VALUES_PER_FIELD + 3;
    with_user_id(UserId::new(ALICE), async {
        for i in 0..over {
            log.record_situation(
                vec!["kb-1".to_string()],
                Situation::new().with(SituationField::Host, format!("host-{i:02}")),
            )
            .await
            .expect("recorded");
        }
    })
    .await;

    let record = situation_of(&log, ALICE, "kb-1").await.expect("a record");
    let hosts: Vec<&str> = record
        .iter()
        .filter(|(field, _)| *field == SituationField::Host)
        .map(|(_, value)| value)
        .collect();
    assert_eq!(
        hosts.len(),
        MAX_SITUATION_VALUES_PER_FIELD,
        "an entry's record must stay bounded: {hosts:?}"
    );
    assert!(
        hosts.contains(&format!("host-{:02}", over - 1).as_str()),
        "the newest value must survive: {hosts:?}"
    );
    assert!(
        !hosts.contains(&"host-00"),
        "the least recently seen value is the one that falls out: {hosts:?}"
    );

    fx.cleanup().await;
}

/// Reaping an entry frees its situation rows with it, so no row can name an
/// entry that does not exist.
#[tokio::test]
async fn deleting_an_entry_takes_its_situation_rows_with_it() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    write_as(&fx.pool, ALICE, "kb-1").await;

    with_user_id(UserId::new(ALICE), async {
        log.record_situation(vec!["kb-1".to_string()], at("workshop", "thursday"))
            .await
            .expect("recorded");
    })
    .await;
    assert!(situation_of(&log, ALICE, "kb-1").await.is_some());

    sqlx::query("DELETE FROM knowledge_base WHERE user_id = $1 AND id = $2")
        .bind(ALICE)
        .bind("kb-1")
        .execute(&fx.pool)
        .await
        .expect("hard delete succeeds");

    let orphans: i64 =
        sqlx::query_scalar("SELECT count(*) FROM knowledge_situation WHERE entry_id = $1")
            .bind("kb-1")
            .fetch_one(&fx.pool)
            .await
            .expect("count succeeds");
    assert_eq!(orphans, 0, "a reaped entry must take its situation with it");

    fx.cleanup().await;
}

/// The cue's own arithmetic, over counts a store really produces, so the
/// storage read and the domain rule are held to one another.
#[tokio::test]
async fn the_cue_the_store_measures_agrees_with_the_domain_rule() {
    let Some(fx) = fixture().await else { return };
    let log = PgKnowledgeUseLog::new(fx.pool.clone());
    let population = SITUATION_MIN_POPULATION as usize + 20;

    for i in 0..population {
        let id = format!("kb-{i}");
        write_as(&fx.pool, ALICE, &id).await;
        let host = if i % 4 == 0 { "workshop" } else { "the-road" };
        with_user_id(UserId::new(ALICE), async {
            log.record_situation(
                vec![id.clone()],
                Situation::new().with(SituationField::Host, host),
            )
            .await
            .expect("recorded");
        })
        .await;
    }

    let measured = with_user_id(UserId::new(ALICE), async {
        log.situation_cue(Situation::new().with(SituationField::Host, "workshop"))
            .await
            .expect("cue read succeeds")
    })
    .await
    .expect("a store this size can grade a cue");

    let at_the_workshop = population.div_ceil(4) as u64;
    let expected = desktop_assistant_core::domain::situation::SituationCue::measured(
        Situation::new().with(SituationField::Host, "workshop"),
        population as u64,
        &BTreeMap::from([(SituationField::Host, at_the_workshop)]),
    )
    .expect("the same counts grade the same cue");

    assert_eq!(
        measured, expected,
        "the store's own counts must produce the cue the domain rule states"
    );

    fx.cleanup().await;
}
