//! The `[Recall]` block's past-turns arm, end to end (#1350).
//!
//! Two stores hold this arm up, and both are exercised here against a real
//! Postgres with the migrations applied: `PgTurnDigestStore::nearest_by_embedding`,
//! which answers with the person's own turns from every conversation they own,
//! and `PgEpisodeUseLog`, which records what the block offered and what the
//! model opened.
//!
//! Each acceptance criterion from the issue is one named test, and each pairs
//! its permit with its refusal: a scope test that only proved a read succeeds
//! would pass against a store with no scoping at all, and an offer test that
//! only proved a row appears would pass against a log that records anything it
//! is handed.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`). Run
//! them against an ephemeral Postgres with `just test-db`.

mod support;

use desktop_assistant_core::ports::embedding::ChunkedEmbedding;
use desktop_assistant_core::ports::episode_use::EpisodeUseLog;
use desktop_assistant_core::ports::knowledge_use::OfferScope;
use desktop_assistant_core::ports::turn_digest::{NewTurnDigest, TurnDigestStore};
use desktop_assistant_storage::{PgEpisodeUseLog, PgTurnDigestStore, UserId, with_user_id};
use sqlx::PgPool;

use support::DbFixture;

/// The model every vector here is stamped with. The store scopes its vector arm
/// to one model, so a test that did not state one would be comparing across
/// dimensions.
const MODEL: &str =
    "nomic-embed-text@1111111111111111111111111111111111111111111111111111111111111111";

/// A conversation row, written directly so a test can choose its owner without
/// going through a request scope.
async fn seed_conversation(pool: &PgPool, user_id: &str, id: &str) {
    sqlx::query("INSERT INTO conversations (id, title, user_id, tags) VALUES ($1, 'test', $2, $3)")
        .bind(id)
        .bind(user_id)
        .bind(Vec::<String>::new())
        .execute(pool)
        .await
        .expect("seed conversation");
}

/// One digest, already carrying `vector`, exactly as an inline-embedded write
/// does.
fn embedded(opening_message_id: &str, content: &str, vector: Vec<f32>) -> NewTurnDigest {
    let mut digest = NewTurnDigest::new(opening_message_id, content);
    digest.embedding = Some(ChunkedEmbedding {
        chunks: vec![vector],
        model: MODEL.to_string(),
    });
    digest
}

/// Acceptance 2 (#1350): a turn from another conversation is offered when it is
/// relevant - and a turn belonging to somebody else never is.
///
/// The permit is the whole point of the store being user-scoped: a turn in one
/// conversation is what answers "when did I last deal with this" in another.
/// The refusal is what makes the permit worth anything, and it is checked at
/// the same read with the same query vector, so a store with no scoping at all
/// would fail here rather than pass.
#[tokio::test]
async fn an_episode_line_from_another_conversation_is_offered_when_it_is_relevant() {
    let Some(fx) = DbFixture::try_new("tdr_scope").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a").await;
    seed_conversation(&fx.pool, "alice", "conv-b").await;
    seed_conversation(&fx.pool, "bob", "conv-bob").await;

    with_user_id(UserId::new("alice"), async {
        store
            .write(
                "conv-a",
                &[embedded(
                    "m-a",
                    "Asked: where did I leave the kustomization",
                    vec![1.0, 0.0, 0.0],
                )],
            )
            .await
            .expect("write in conversation a");
    })
    .await;
    with_user_id(UserId::new("bob"), async {
        store
            .write(
                "conv-bob",
                &[embedded(
                    "m-bob",
                    "Asked: where did I leave the kustomization",
                    vec![1.0, 0.0, 0.0],
                )],
            )
            .await
            .expect("write bob's own turn");
    })
    .await;

    with_user_id(UserId::new("alice"), async {
        // Alice is working in conversation b, and the read carries no
        // conversation of its own: the store is hers, not one conversation's.
        let found = store
            .nearest_by_embedding(vec![1.0, 0.0, 0.0], MODEL, 10)
            .await
            .expect("the read answers");
        let openings: Vec<&str> = found
            .digests
            .iter()
            .map(|(d, _)| d.opening_message_id.as_str())
            .collect();

        assert!(
            openings.contains(&"m-a"),
            "a turn from another conversation of hers must be reachable, got {openings:?}"
        );
        assert!(
            !openings.contains(&"m-bob"),
            "bob's turn is at the same distance from the same query and must never be \
             offered to alice, got {openings:?}"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance 4 (#1350): an offered episode is recorded in the use log.
///
/// Paired with two refusals in the same test, because an offer write that
/// recorded whatever it was handed would pass a permit-only check. An id
/// belonging to somebody else records nothing, and an id no digest carries
/// records nothing.
#[tokio::test]
async fn an_offered_episode_is_recorded_in_the_use_log() {
    let Some(fx) = DbFixture::try_new("tdr_offer").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());
    let log = PgEpisodeUseLog::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a").await;
    seed_conversation(&fx.pool, "bob", "conv-bob").await;

    let bobs_id = with_user_id(UserId::new("bob"), async {
        store
            .write(
                "conv-bob",
                &[NewTurnDigest::new("m-bob", "Asked: bob's own")],
            )
            .await
            .expect("write bob's turn")
            .remove(0)
            .id
    })
    .await;

    with_user_id(UserId::new("alice"), async {
        let mine = store
            .write("conv-a", &[NewTurnDigest::new("m-a", "Asked: mine")])
            .await
            .expect("write")
            .remove(0)
            .id;

        let written = log
            .record_offered(
                OfferScope::recall("conv-a"),
                vec![mine.clone(), bobs_id.clone(), "no-such-id".to_string()],
            )
            .await
            .expect("the offer write answers");
        assert_eq!(
            written, 1,
            "only the id this person owns is an offer; the other two record nothing"
        );

        let records = log
            .records(vec![
                mine.clone(),
                bobs_id.clone(),
                "no-such-id".to_string(),
            ])
            .await
            .expect("the read answers");
        assert_eq!(records.len(), 1, "one record, for one offer");
        assert_eq!(records[0].entry_id, mine);
        assert_eq!(records[0].offered_count, 1);
        assert_eq!(
            records[0].opened_count, 0,
            "surfaced is not opened, and the ranking reads the difference"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Acceptance 5 (#1350): an opened episode is recorded in the use log.
///
/// Paired with its refusal: a read of an episode nothing offered in this
/// conversation records no open. That rule is what makes an open evidence that
/// the block worked rather than ordinary bookkeeping - and taking the offer
/// down is what makes a retried tool call idempotent, which is checked here
/// too.
#[tokio::test]
async fn an_opened_episode_is_recorded_in_the_use_log() {
    let Some(fx) = DbFixture::try_new("tdr_open").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());
    let log = PgEpisodeUseLog::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a").await;

    with_user_id(UserId::new("alice"), async {
        let written = store
            .write(
                "conv-a",
                &[
                    NewTurnDigest::new("m-offered", "Asked: the one the block offered"),
                    NewTurnDigest::new("m-unoffered", "Asked: the one nothing offered"),
                ],
            )
            .await
            .expect("write");
        let offered_id = written
            .iter()
            .find(|d| d.opening_message_id == "m-offered")
            .expect("the offered digest")
            .id
            .clone();
        let unoffered_id = written
            .iter()
            .find(|d| d.opening_message_id == "m-unoffered")
            .expect("the unoffered digest")
            .id
            .clone();

        log.record_offered(OfferScope::recall("conv-a"), vec![offered_id.clone()])
            .await
            .expect("the block's offer");

        let opens = log
            .record_opened(
                "conv-a".to_string(),
                vec![offered_id.clone(), unoffered_id.clone()],
            )
            .await
            .expect("the open write answers");
        assert_eq!(
            opens, 1,
            "only the standing offer becomes an open; a read nothing offered is not evidence"
        );

        // A retried tool call reads the same episode again. The offer is
        // already down, so nothing is added.
        let again = log
            .record_opened("conv-a".to_string(), vec![offered_id.clone()])
            .await
            .expect("the repeat answers");
        assert_eq!(again, 0, "a second read of the same turn is one open");

        let records = log
            .records(vec![offered_id.clone(), unoffered_id.clone()])
            .await
            .expect("the read answers");
        let opened = records
            .iter()
            .find(|r| r.entry_id == offered_id)
            .expect("a record for the opened episode");
        assert_eq!(opened.offered_count, 1);
        assert_eq!(opened.opened_count, 1);
        assert_eq!(
            opened.recent_uses.len(),
            1,
            "the open is in the recent-use window the activation score reads"
        );
        assert!(
            records.iter().all(|r| r.entry_id != unoffered_id),
            "an episode nothing offered has no record at all"
        );
    })
    .await;

    fx.cleanup().await;
}

/// Deleting a conversation deletes its digests (#1349), and the use rows those
/// digests carried go with them.
///
/// Without the cascade a person who deleted a conversation would leave counters
/// behind naming turns that no longer exist - and the ranking would go on
/// reading them.
#[tokio::test]
async fn deleting_a_conversation_frees_the_use_rows_of_its_episodes() {
    let Some(fx) = DbFixture::try_new("tdr_cascade").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());
    let log = PgEpisodeUseLog::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a").await;

    let id = with_user_id(UserId::new("alice"), async {
        let id = store
            .write("conv-a", &[NewTurnDigest::new("m-a", "Asked: mine")])
            .await
            .expect("write")
            .remove(0)
            .id;
        log.record_offered(OfferScope::recall("conv-a"), vec![id.clone()])
            .await
            .expect("offer");
        log.record_opened("conv-a".to_string(), vec![id.clone()])
            .await
            .expect("open");
        id
    })
    .await;

    let before: i64 =
        sqlx::query_scalar("SELECT count(*) FROM episode_use_stats WHERE episode_id = $1")
            .bind(&id)
            .fetch_one(&fx.pool)
            .await
            .expect("count stats");
    assert_eq!(before, 1, "the use row exists before the delete");

    sqlx::query("DELETE FROM conversations WHERE id = $1")
        .bind("conv-a")
        .execute(&fx.pool)
        .await
        .expect("delete the conversation");

    let after: i64 =
        sqlx::query_scalar("SELECT count(*) FROM episode_use_stats WHERE episode_id = $1")
            .bind(&id)
            .fetch_one(&fx.pool)
            .await
            .expect("count stats");
    assert_eq!(after, 0, "the use rows went with the digest");

    fx.cleanup().await;
}

/// The read honours disposition the way `knowledge_search` does: `obsolete` is
/// left out, and every other value comes back so a refuted turn stays findable
/// when the prompt is about its subject.
///
/// The permit and the refusal are one read against one query vector, so a query
/// that filtered nothing, or filtered everything, fails.
#[tokio::test]
async fn the_episode_read_leaves_out_an_obsolete_turn_and_keeps_a_refuted_one() {
    use desktop_assistant_core::domain::Disposition;

    let Some(fx) = DbFixture::try_new("tdr_disp").await else {
        return;
    };
    let store = PgTurnDigestStore::new(fx.pool.clone());

    seed_conversation(&fx.pool, "alice", "conv-a").await;

    with_user_id(UserId::new("alice"), async {
        let written = store
            .write(
                "conv-a",
                &[
                    embedded("m-refuted", "Asked: the refuted one", vec![1.0, 0.0, 0.0]),
                    embedded("m-obsolete", "Asked: the obsolete one", vec![1.0, 0.0, 0.0]),
                    embedded("m-active", "Asked: the active one", vec![1.0, 0.0, 0.0]),
                ],
            )
            .await
            .expect("write");
        let id_of = |opening: &str| {
            written
                .iter()
                .find(|d| d.opening_message_id == opening)
                .expect("a written digest")
                .id
                .clone()
        };
        store
            .set_disposition(&id_of("m-refuted"), Disposition::Refuted, None, None)
            .await
            .expect("refute");
        store
            .set_disposition(&id_of("m-obsolete"), Disposition::Obsolete, None, None)
            .await
            .expect("retire");

        let found = store
            .nearest_by_embedding(vec![1.0, 0.0, 0.0], MODEL, 10)
            .await
            .expect("the read answers");
        let openings: Vec<&str> = found
            .digests
            .iter()
            .map(|(d, _)| d.opening_message_id.as_str())
            .collect();

        assert!(openings.contains(&"m-active"), "{openings:?}");
        assert!(
            openings.contains(&"m-refuted"),
            "a refuted turn stays findable, marked, so a correction can be reported rather \
             than silently forgotten: {openings:?}"
        );
        assert!(
            !openings.contains(&"m-obsolete"),
            "an obsolete turn no longer applies and is not offered: {openings:?}"
        );
    })
    .await;

    fx.cleanup().await;
}
