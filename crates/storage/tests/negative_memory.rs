//! Negative memory (#1126): what the store owns of "learning from being
//! burned".
//!
//! The rule a burn follows is pure and is tested beside it, in
//! `core::domain::negative_memory`. What lives here cannot be proved without a
//! real database: the partial unique index that lets a correction share its
//! original's identity, the confirm-or-write branch, the facet cascade, the
//! reap, and the tenant boundary.
//!
//! Each acceptance criterion the storage layer owns is one named test below:
//!
//! - `a_single_failed_outcome_writes_a_burn_at_full_strength`
//! - `a_burn_records_the_action_the_context_and_the_outcome`
//! - `broadening_a_burns_scope_needs_a_second_occurrence`
//! - `a_second_failure_with_a_different_argument_is_a_second_burn`
//! - `extinction_writes_an_overlay_and_the_original_remains_readable`
//! - `extinguishing_a_burn_twice_changes_nothing`
//! - `a_new_burn_can_be_written_after_the_old_one_was_extinguished`
//! - `a_second_live_burn_with_one_identity_is_refused_by_the_index`
//! - `a_burn_scoped_by_a_facet_this_build_cannot_name_is_not_returned`
//! - `a_burn_and_the_correction_over_it_are_reaped_together`
//! - `a_burn_nothing_has_confirmed_is_reaped_on_the_next_write`
//! - `a_reaped_burn_takes_its_facets_with_it`
//! - `a_cross_tenant_read_of_a_negative_memory_returns_nothing`
//! - `extinguishing_another_users_burn_changes_nothing`
//! - `a_negative_mark_does_not_write_a_negative_memory`
//! - `writing_a_negative_memory_does_not_mark_a_knowledge_entry`
//! - `the_live_read_is_bounded`
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test negative_memory
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::negative_memory::{
    FORGET_DAYS, FULL_STRENGTH, Facet, MAX_LIVE_BURNS, NegativeMemory, NegativeMemoryKind, Scope,
    fingerprint,
};
use desktop_assistant_core::domain::{KnowledgeEntry, MarkPolarity, MarkSource, SituationField};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::knowledge_use::{KnowledgeUseLog, MarkRequest};
use desktop_assistant_core::ports::negative_memory::{
    BurnObservation, BurnWrite, NegativeMemoryStore,
};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{
    PgKnowledgeBaseStore, PgKnowledgeUseLog, PgNegativeMemoryStore, UserId, with_user_id,
};
use sqlx::PgPool;

const ALICE: &str = "burn-alice";
const BOB: &str = "burn-bob";
const ACTION: &str = "terminal_run";

async fn fixture() -> Option<support::DbFixture> {
    support::DbFixture::try_new("negmem1126").await
}

/// The scope the tests are about: clearing a build directory, at the workshop,
/// on a Thursday morning.
fn at_the_workshop() -> Scope {
    Scope::new()
        .with(Facet::Argument("command".to_string()), "rm -rf build")
        .with(Facet::Argument("cwd".to_string()), "/srv/app")
        .with(Facet::Situation(SituationField::Host), "workshop")
        .with(Facet::Situation(SituationField::TimeOfDay), "morning")
        .with(Facet::Situation(SituationField::Weekday), "thursday")
}

/// The same act, on another host, on another day.
fn on_a_laptop() -> Scope {
    Scope::new()
        .with(Facet::Argument("command".to_string()), "rm -rf build")
        .with(Facet::Argument("cwd".to_string()), "/srv/app")
        .with(Facet::Situation(SituationField::Host), "laptop")
        .with(Facet::Situation(SituationField::TimeOfDay), "evening")
        .with(Facet::Situation(SituationField::Weekday), "sunday")
}

/// One observation of `scope` going badly. `act` names the call, so two
/// observations with different acts are two lessons however alike their
/// recorded facets look.
fn observation_of(act: &str, scope: Scope, outcome: &str) -> BurnObservation {
    BurnObservation {
        action: ACTION.to_string(),
        fingerprint: fingerprint(&serde_json::json!({ "call": act })),
        scope,
        outcome: outcome.to_string(),
    }
}

/// The act every test is about, unless it says otherwise.
fn observation(scope: Scope, outcome: &str) -> BurnObservation {
    observation_of("rm -rf build in /srv/app", scope, outcome)
}

async fn burn_as(store: &PgNegativeMemoryStore, user: &str, obs: BurnObservation) -> BurnWrite {
    with_user_id(UserId::new(user), async { store.record_burn(obs).await })
        .await
        .expect("recording a bad outcome succeeds")
}

async fn live_as(store: &PgNegativeMemoryStore, user: &str) -> Vec<NegativeMemory> {
    with_user_id(UserId::new(user), async { store.live_burns().await })
        .await
        .expect("reading the live burns succeeds")
}

async fn history_as(store: &PgNegativeMemoryStore, user: &str) -> Vec<NegativeMemory> {
    with_user_id(UserId::new(user), async {
        store.history(ACTION.to_string()).await
    })
    .await
    .expect("reading the history succeeds")
}

async fn extinguish_as(
    store: &PgNegativeMemoryStore,
    user: &str,
    ids: Vec<String>,
    note: &str,
) -> Vec<String> {
    with_user_id(UserId::new(user), async {
        store.extinguish(ids, note.to_string()).await
    })
    .await
    .expect("writing a correction succeeds")
}

/// Backdate a burn's last confirmation, so the decay and reap rules can be
/// exercised without waiting weeks.
async fn backdate(pool: &PgPool, id: &str, days: i32) {
    sqlx::query(
        "UPDATE negative_memory \
         SET last_confirmed_at = NOW() - make_interval(days => $2) \
         WHERE id = $1",
    )
    .bind(id)
    .bind(days)
    .execute(pool)
    .await
    .expect("backdate the confirmation stamp");
}

async fn count(pool: &PgPool, sql: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(sqlx::AssertSqlSafe(sql.to_string()))
        .fetch_one(pool)
        .await
        .expect("count query succeeds")
}

// --- One trial is enough ---------------------------------------------------

/// Acceptance (#1126): one bad outcome writes a burn, and it holds full
/// strength straight away. Nothing has to happen twice.
#[tokio::test]
async fn a_single_failed_outcome_writes_a_burn_at_full_strength() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let write = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    assert_eq!(write.occurrences, 1, "the first bad outcome writes the row");
    assert_eq!(
        write.widened_by, 0,
        "a first write widens nothing, because there is nothing to widen"
    );

    let live = live_as(&store, ALICE).await;
    assert_eq!(live.len(), 1);
    assert!(
        (live[0].strength(chrono::Utc::now()) - FULL_STRENGTH).abs() < 1e-6,
        "a burn is written at full strength, not built up to it"
    );
    fx.cleanup().await;
}

/// Acceptance (#1126): a stored burn carries the action, the context - its own
/// arguments and the situation it happened in - and what went wrong.
#[tokio::test]
async fn a_burn_records_the_action_the_context_and_the_outcome() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;

    let live = live_as(&store, ALICE).await;
    let held = &live[0];
    assert_eq!(held.action, ACTION);
    assert_eq!(held.kind, NegativeMemoryKind::Burn);
    assert_eq!(held.outcome, "build is a mount point");
    assert_eq!(
        held.scope,
        at_the_workshop(),
        "every observed facet survives the round trip, arguments and situation alike"
    );
    fx.cleanup().await;
}

// --- Broadening needs a second occurrence -----------------------------------

/// Acceptance (#1126): the scope written by one occurrence stays as it was
/// written. A second occurrence somewhere else drops the situation facets it
/// disagrees with, and only then does the burn reach beyond where it was born.
#[tokio::test]
async fn broadening_a_burns_scope_needs_a_second_occurrence() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let first = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    assert_eq!(
        live_as(&store, ALICE).await[0].scope,
        at_the_workshop(),
        "one occurrence keeps the scope it was written with"
    );

    let second = burn_as(
        &store,
        ALICE,
        observation(on_a_laptop(), "build is a mount point here too"),
    )
    .await;
    assert_eq!(second.id, first.id, "the same act is the same lesson");
    assert_eq!(second.occurrences, 2);
    assert_eq!(
        second.widened_by, 3,
        "the host, the part of day and the weekday all disagreed, so all three drop"
    );

    let widened = &live_as(&store, ALICE).await[0];
    assert_eq!(
        widened.scope,
        Scope::new()
            .with(Facet::Argument("command".to_string()), "rm -rf build")
            .with(Facet::Argument("cwd".to_string()), "/srv/app"),
        "what is left is the act itself, which is what both occurrences shared"
    );
    fx.cleanup().await;
}

/// The identity guard, in the store: a failure of the same tool with a
/// different argument is a second lesson, and it leaves the first one's scope
/// exactly as it was.
#[tokio::test]
async fn a_second_failure_with_a_different_argument_is_a_second_burn() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let first = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    let elsewhere = at_the_workshop().with(Facet::Argument("cwd".to_string()), "/srv/other");
    let second = burn_as(
        &store,
        ALICE,
        observation_of("rm -rf build in /srv/other", elsewhere, "no such directory"),
    )
    .await;

    assert_ne!(second.id, first.id, "another directory is another lesson");
    let live = live_as(&store, ALICE).await;
    assert_eq!(live.len(), 2);
    let original = live
        .iter()
        .find(|m| m.id == first.id)
        .expect("the first lesson is still held");
    assert_eq!(
        original.scope,
        at_the_workshop(),
        "a separate lesson cannot widen this one"
    );
    fx.cleanup().await;
}

// --- Extinction is an overlay -----------------------------------------------

/// Acceptance (#1126): extinction writes a correction over the burn, and the
/// original stays readable with everything it held.
#[tokio::test]
async fn extinction_writes_an_overlay_and_the_original_remains_readable() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let burn = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    let extinguished = extinguish_as(
        &store,
        ALICE,
        vec![burn.id.clone()],
        "the same call succeeded; the mount point was removed",
    )
    .await;
    assert_eq!(extinguished, vec![burn.id.clone()]);

    assert!(
        live_as(&store, ALICE).await.is_empty(),
        "an extinguished burn is not live"
    );

    let history = history_as(&store, ALICE).await;
    assert_eq!(history.len(), 2, "the burn and the correction over it");
    let original = history
        .iter()
        .find(|m| m.id == burn.id)
        .expect("the original survives its own extinction");
    assert_eq!(
        original.outcome, "build is a mount point",
        "the lesson is still there to read"
    );
    assert_eq!(
        original.scope,
        at_the_workshop(),
        "and so is everything it was scoped to"
    );
    let correction_id = original
        .superseded_by
        .clone()
        .expect("the original names what corrected it");
    let correction = history
        .iter()
        .find(|m| m.id == correction_id)
        .expect("the correction is a row of its own");
    assert_eq!(correction.kind, NegativeMemoryKind::Correction);
    assert!(correction.outcome.contains("succeeded"));
    assert_eq!(
        correction.scope,
        at_the_workshop(),
        "the correction says which lesson it corrects, in the lesson's own terms"
    );
    fx.cleanup().await;
}

/// Extinction is idempotent: a repeat writes no second correction and reports
/// nothing extinguished.
#[tokio::test]
async fn extinguishing_a_burn_twice_changes_nothing() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let burn = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    extinguish_as(&store, ALICE, vec![burn.id.clone()], "it works now").await;
    let again = extinguish_as(&store, ALICE, vec![burn.id.clone()], "it works now").await;

    assert!(again.is_empty(), "there was nothing left to extinguish");
    assert_eq!(history_as(&store, ALICE).await.len(), 2);
    fx.cleanup().await;
}

/// The partial unique index earns its predicate: once a lesson is corrected,
/// the same act failing again is a fresh lesson beside the record of the old
/// one, not a write that collides with it.
#[tokio::test]
async fn a_new_burn_can_be_written_after_the_old_one_was_extinguished() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let first = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    extinguish_as(&store, ALICE, vec![first.id.clone()], "it works now").await;

    let second = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "it is a mount point again"),
    )
    .await;
    assert_ne!(
        second.id, first.id,
        "a fresh lesson, not the old one revived"
    );
    assert_eq!(second.occurrences, 1, "and it starts its own count");

    let live = live_as(&store, ALICE).await;
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].id, second.id);
    assert_eq!(
        history_as(&store, ALICE).await.len(),
        3,
        "the old burn, its correction, and the new burn"
    );
    fx.cleanup().await;
}

/// The partial unique index is the backstop the confirm branch never reaches:
/// nothing in the adapter tries to write a second live lesson for one identity,
/// so only the index itself holds the "one live lesson" rule against a race or
/// a later caller. Proved by attempting the write the index exists to refuse.
#[tokio::test]
async fn a_second_live_burn_with_one_identity_is_refused_by_the_index() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let first = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    let fingerprint =
        sqlx::query_scalar::<_, String>("SELECT fingerprint FROM negative_memory WHERE id = $1")
            .bind(&first.id)
            .fetch_one(&fx.pool)
            .await
            .expect("read the identity back");

    let duplicate = sqlx::query(
        "INSERT INTO negative_memory (id, user_id, action, fingerprint, kind, outcome) \
         VALUES ($1, $2, $3, $4, 'burn', $5)",
    )
    .bind("nm-duplicate")
    .bind(ALICE)
    .bind(ACTION)
    .bind(&fingerprint)
    .bind("the same lesson, written twice")
    .execute(&fx.pool)
    .await;
    assert!(
        duplicate.is_err(),
        "the index must refuse a second live lesson for one identity"
    );

    // The same row is accepted once the first is extinguished, which is the
    // other half of what the predicate buys.
    extinguish_as(&store, ALICE, vec![first.id.clone()], "it works now").await;
    sqlx::query(
        "INSERT INTO negative_memory (id, user_id, action, fingerprint, kind, outcome) \
         VALUES ($1, $2, $3, $4, 'burn', $5)",
    )
    .bind("nm-duplicate")
    .bind(ALICE)
    .bind(ACTION)
    .bind(&fingerprint)
    .bind("it went badly again")
    .execute(&fx.pool)
    .await
    .expect("a fresh lesson may take the identity a correction released");

    fx.cleanup().await;
}

// --- The writer is the reaper ------------------------------------------------

/// A burn nothing has confirmed past the forget horizon is dropped on the next
/// write, which is what bounds the table without a sweep.
#[tokio::test]
async fn a_burn_nothing_has_confirmed_is_reaped_on_the_next_write() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let old = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    backdate(&fx.pool, &old.id, FORGET_DAYS.round() as i32 + 1).await;

    let elsewhere = at_the_workshop().with(Facet::Argument("cwd".to_string()), "/srv/other");
    burn_as(
        &store,
        ALICE,
        observation_of("rm -rf build in /srv/other", elsewhere, "no such directory"),
    )
    .await;

    let live = live_as(&store, ALICE).await;
    assert_eq!(live.len(), 1, "only the fresh lesson is left");
    assert_ne!(live[0].id, old.id);
    fx.cleanup().await;
}

/// A reaped burn takes its facets with it, so the facet table cannot outlive
/// what it describes.
#[tokio::test]
async fn a_reaped_burn_takes_its_facets_with_it() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let old = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    assert_eq!(
        count(&fx.pool, "SELECT count(*) FROM negative_memory_facet").await,
        5,
        "two arguments and three situation values"
    );
    backdate(&fx.pool, &old.id, FORGET_DAYS.round() as i32 + 1).await;

    let elsewhere = at_the_workshop().with(Facet::Argument("cwd".to_string()), "/srv/other");
    burn_as(
        &store,
        ALICE,
        observation_of("rm -rf build in /srv/other", elsewhere, "no such directory"),
    )
    .await;

    assert_eq!(
        count(&fx.pool, "SELECT count(*) FROM negative_memory_facet").await,
        5,
        "the reaped lesson's facets went with it, leaving only the new lesson's"
    );
    fx.cleanup().await;
}

/// The per-turn read is bounded, so a long history cannot turn one read into an
/// unbounded one.
#[tokio::test]
async fn the_live_read_is_bounded() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    for i in 0..MAX_LIVE_BURNS + 5 {
        let scope = Scope::new().with(Facet::Argument("cwd".to_string()), format!("/srv/{i}"));
        burn_as(
            &store,
            ALICE,
            observation_of(&format!("clean /srv/{i}"), scope, "no such directory"),
        )
        .await;
    }
    assert_eq!(live_as(&store, ALICE).await.len(), MAX_LIVE_BURNS);
    fx.cleanup().await;
}

/// A row scoped by a dimension this build does not know is not returned at all.
///
/// Dropping the unreadable facet and keeping the row is the tempting answer and
/// the dangerous one: the burn would lose a requirement and fire on acts it had
/// never been seen with. The case is ordinary - a database written by a build
/// that knows one more situation dimension, read by one that does not, is a
/// rollback.
#[tokio::test]
async fn a_burn_scoped_by_a_facet_this_build_cannot_name_is_not_returned() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let burn = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    assert_eq!(live_as(&store, ALICE).await.len(), 1);

    sqlx::query(
        "INSERT INTO negative_memory_facet (user_id, memory_id, kind, name, value) \
         VALUES ($1, $2, 'situation', 'moon_phase', 'waxing')",
    )
    .bind(ALICE)
    .bind(&burn.id)
    .execute(&fx.pool)
    .await
    .expect("a later build's dimension lands in the table");

    assert!(
        live_as(&store, ALICE).await.is_empty(),
        "a lesson this build cannot read whole is a lesson it must not act on"
    );
    fx.cleanup().await;
}

/// A burn and the correction over it are one unit and are reaped together.
///
/// A burn is always older than what corrected it, so reaping on each row's own
/// stamp would take the burn first and leave a correction naming nothing - a
/// row saying an unnamed lesson stopped applying.
#[tokio::test]
async fn a_burn_and_the_correction_over_it_are_reaped_together() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let burn = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    extinguish_as(&store, ALICE, vec![burn.id.clone()], "it works now").await;
    // The burn is well past the horizon; its correction is not.
    backdate(&fx.pool, &burn.id, FORGET_DAYS.round() as i32 + 1).await;

    burn_as(
        &store,
        ALICE,
        observation_of(
            "clean /srv/other",
            Scope::new().with(Facet::Argument("cwd".to_string()), "/srv/other"),
            "no such directory",
        ),
    )
    .await;

    let correction_id =
        sqlx::query_scalar::<_, String>("SELECT superseded_by FROM negative_memory WHERE id = $1")
            .bind(&burn.id)
            .fetch_one(&fx.pool)
            .await
            .expect("the burn names its correction");

    let held = history_as(&store, ALICE).await;
    assert!(
        held.iter().any(|m| m.id == burn.id) && held.iter().any(|m| m.id == correction_id),
        "the pair survives while its correction is still fresh"
    );
    backdate(&fx.pool, &correction_id, FORGET_DAYS.round() as i32 + 1).await;

    burn_as(
        &store,
        ALICE,
        observation_of(
            "clean /srv/third",
            Scope::new().with(Facet::Argument("cwd".to_string()), "/srv/third"),
            "no such directory",
        ),
    )
    .await;

    let held = history_as(&store, ALICE).await;
    assert!(
        !held.iter().any(|m| m.id == burn.id) && !held.iter().any(|m| m.id == correction_id),
        "once both are past the horizon the pair goes together, and neither is \
         left naming the other"
    );
    fx.cleanup().await;
}

// --- The tenant boundary ------------------------------------------------------

/// One person's lessons are not another's. What a person's assistant tried and
/// how it failed says as much about the person as the work did.
#[tokio::test]
async fn a_cross_tenant_read_of_a_negative_memory_returns_nothing() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;

    assert!(live_as(&store, BOB).await.is_empty());
    assert!(history_as(&store, BOB).await.is_empty());
    assert_eq!(live_as(&store, ALICE).await.len(), 1);
    fx.cleanup().await;
}

/// Nor can one person correct another's lesson, even holding its id.
#[tokio::test]
async fn extinguishing_another_users_burn_changes_nothing() {
    let Some(fx) = fixture().await else { return };
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    let burn = burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;
    let claimed = extinguish_as(&store, BOB, vec![burn.id.clone()], "it works for me").await;

    assert!(claimed.is_empty());
    let live = live_as(&store, ALICE).await;
    assert_eq!(live.len(), 1, "the lesson is untouched");
    assert_eq!(live[0].superseded_by, None);
    fx.cleanup().await;
}

// --- Three negatives, and they stay apart --------------------------------------

/// Acceptance (#1126): marking a knowledge entry useless is #698's mechanism
/// and says something about an entry. It writes no negative memory, because a
/// burn is about an act.
#[tokio::test]
async fn a_negative_mark_does_not_write_a_negative_memory() {
    let Some(fx) = fixture().await else { return };
    let entries = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    let uses = PgKnowledgeUseLog::new(fx.pool.clone());
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    with_user_id(UserId::new(ALICE), async {
        entries
            .write(KnowledgeEntry::new(
                "kb-1",
                "a fact that turned out wrong",
                vec![],
            ))
            .await
            .expect("write the entry");
        let marked = uses
            .record_mark(MarkRequest {
                entry_ids: vec!["kb-1".to_string()],
                polarity: MarkPolarity::Negative,
                source: MarkSource::Model,
                reason: Some("this was wrong".to_string()),
            })
            .await
            .expect("mark the entry");
        assert_eq!(marked, vec!["kb-1".to_string()], "the mark did land");
    })
    .await;

    assert!(
        live_as(&store, ALICE).await.is_empty(),
        "a mark on an entry is not a lesson about an action"
    );
    assert_eq!(
        count(&fx.pool, "SELECT count(*) FROM negative_memory").await,
        0
    );
    fx.cleanup().await;
}

/// And the other way: recording a burn leaves the use log alone. The two answer
/// different questions about different objects.
#[tokio::test]
async fn writing_a_negative_memory_does_not_mark_a_knowledge_entry() {
    let Some(fx) = fixture().await else { return };
    let entries = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    let store = PgNegativeMemoryStore::new(fx.pool.clone());

    with_user_id(UserId::new(ALICE), async {
        entries
            .write(KnowledgeEntry::new("kb-1", "a fact worth keeping", vec![]))
            .await
            .expect("write the entry");
    })
    .await;

    burn_as(
        &store,
        ALICE,
        observation(at_the_workshop(), "build is a mount point"),
    )
    .await;

    assert_eq!(
        count(&fx.pool, "SELECT count(*) FROM knowledge_use_marks").await,
        0,
        "a burn marks nothing"
    );
    assert_eq!(
        count(&fx.pool, "SELECT count(*) FROM knowledge_use_stats").await,
        0
    );
    fx.cleanup().await;
}
