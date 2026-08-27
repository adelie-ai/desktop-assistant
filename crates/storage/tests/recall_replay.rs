//! The recall snapshot, the labelled set, and the replay (#1328).
//!
//! Retrieval ranks knowledge entries by activation, and nothing before this
//! could say whether a change to that score helped: background writers move
//! the corpus underneath any two measurements, and no query carried a
//! recorded correct answer. This suite pins the properties that make
//! `desktop_assistant_storage::recall_replay` a trustworthy instrument
//! rather than a second guess:
//!
//! - a snapshot's identity names the embedding model it was taken under, and
//!   a row embedded under a different model is excluded and counted, never
//!   silently absorbed;
//! - a replay is deterministic (frozen `now`, cached query embeddings) and
//!   refuses to compare vectors from two different embedding spaces;
//! - a case whose expected entry vanished from the snapshot is reported, not
//!   dropped, and the report never truncates the case list silently;
//! - the ranked scan over the frozen tables orders a seeded corpus exactly
//!   as the live scan does, because it reads distance through the same SQL
//!   expression;
//! - an untraceable case (no source turn, no note) is refused, and a
//!   snapshot a case still depends on cannot be dropped out from under it.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test recall_replay
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use chrono::Utc;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{
    CaseInput, CaseOutcome, PgKnowledgeBaseStore, PgPool, UserId, add_case, cache_case_embedding,
    drop_snapshot, ranked_snapshot_scan, run_replay, set_case_baseline, take_snapshot,
    with_user_id,
};
use pgvector::Vector;

const USER: &str = "recall-replay-user";
const MODEL_A: &str = "replay-test-model-a";
const MODEL_B: &str = "replay-test-model-b";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("recallreplay").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

/// A three-dimensional unit vector `radians` around from the first axis, so a
/// fixture can seed a spread of distances rather than only "same" and
/// "orthogonal". Bland by construction: nothing here is a distance of zero
/// from anything else it is compared against.
fn at_angle(radians: f32) -> Vec<f32> {
    vec![radians.cos(), radians.sin(), 0.0]
}

/// Write a knowledge entry as `user` and stamp it with an embedding under
/// `model`, exactly the way `recall_candidates.rs` does — writes never embed
/// inline, so a test stands in for the background backfill. Content is
/// deliberately bland ("a stored fact N") and carries no `source`, so
/// salience reads empty for every row and activation collapses to the
/// semantic term alone — the property the distance-parity test needs to
/// compare a plain distance order against.
async fn seed_entry(pool: &PgPool, id: &str, chunk: Vec<f32>, model: &str) {
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());
    with_user_id(UserId::new(USER), async {
        store
            .write(KnowledgeEntry::new(
                id,
                format!("a stored fact, entry {id}"),
                vec!["topic".to_string()],
            ))
            .await
            .expect("write succeeds");
    })
    .await;

    let vectors: Vec<Vector> = vec![Vector::from(chunk)];
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = $3, embeddings_updated_at = NOW() \
         WHERE id = $2",
    )
    .bind(&vectors)
    .bind(id)
    .bind(model)
    .execute(pool)
    .await
    .expect("stamp embedding");
}

/// Stamp a knowledge entry's disposition directly, the way consolidation
/// would after judging it wrong, stale or redundant. `disposition` is the
/// stored spelling (`"superseded"`, `"obsolete"`, `"trivial"`, ...).
async fn seed_disposition(pool: &PgPool, id: &str, disposition: &str) {
    sqlx::query("UPDATE knowledge_base SET disposition = $1 WHERE id = $2")
        .bind(disposition)
        .bind(id)
        .execute(pool)
        .await
        .expect("stamp disposition");
}

/// Add a case with a note (always traceable) and cache its query embedding
/// under `model`, so a test can drive `run_replay` without a live embedder.
async fn seed_case(
    pool: &PgPool,
    expected_entry_id: &str,
    query_embedding: Vec<f32>,
    model: &str,
) -> String {
    let case_id = add_case(
        pool,
        USER,
        CaseInput {
            query_text: "what stored fact matches this?",
            expected_entry_id,
            source_request_id: None,
            note: Some("seeded for the recall_replay DB suite"),
            baseline_snapshot_id: None,
        },
    )
    .await
    .expect("add_case succeeds for a traceable case");
    cache_case_embedding(pool, USER, &case_id, model, query_embedding)
        .await
        .expect("cache the case's query embedding");
    case_id
}

#[tokio::test]
async fn a_snapshot_records_the_embedding_model_it_was_taken_under() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-1", at_angle(0.0), MODEL_A).await;
    seed_entry(&fx.pool, "kb-2", at_angle(0.3), MODEL_A).await;

    let manifest = take_snapshot(&fx.pool, USER, "snap-a")
        .await
        .expect("take_snapshot succeeds");

    assert_eq!(manifest.embedding_model, MODEL_A);
    assert_eq!(manifest.entry_count, 2);
    assert_eq!(manifest.excluded_count, 0);

    fx.cleanup().await;
}

#[tokio::test]
async fn a_snapshot_excludes_rows_embedded_under_another_model_and_reports_the_count() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-1", at_angle(0.0), MODEL_A).await;
    seed_entry(&fx.pool, "kb-2", at_angle(0.3), MODEL_A).await;
    seed_entry(&fx.pool, "kb-other", at_angle(0.6), MODEL_B).await;

    let manifest = take_snapshot(&fx.pool, USER, "snap-mixed")
        .await
        .expect("take_snapshot succeeds");

    assert_eq!(manifest.embedding_model, MODEL_A, "the majority model wins");
    assert_eq!(
        manifest.entry_count, 2,
        "the minority-model row is excluded"
    );
    assert_eq!(
        manifest.excluded_count, 1,
        "the exclusion is counted, not silently dropped"
    );

    let copied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM recall_snapshot_entries \
         WHERE snapshot_id = $1 AND entry_id = 'kb-other'",
    )
    .bind(&manifest.id)
    .fetch_one(&fx.pool)
    .await
    .expect("count copied rows");
    assert_eq!(copied, 0, "the excluded row must not be in the snapshot");

    fx.cleanup().await;
}

#[tokio::test]
async fn replaying_a_set_against_a_snapshot_twice_gives_identical_ranks() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-near", at_angle(0.0), MODEL_A).await;
    seed_entry(&fx.pool, "kb-mid", at_angle(0.4), MODEL_A).await;
    seed_entry(&fx.pool, "kb-far", at_angle(1.2), MODEL_A).await;

    let manifest = take_snapshot(&fx.pool, USER, "snap-twice")
        .await
        .expect("take_snapshot succeeds");
    seed_case(&fx.pool, "kb-mid", at_angle(0.4), MODEL_A).await;

    let first = run_replay(&fx.pool, USER, &manifest)
        .await
        .expect("first replay succeeds");
    let second = run_replay(&fx.pool, USER, &manifest)
        .await
        .expect("second replay succeeds");

    let rank_of =
        |report: &desktop_assistant_storage::ReplayReport| match &report.results[0].outcome {
            CaseOutcome::Ranked { rank, .. } => *rank,
            CaseOutcome::ExpectedEntryMissing { .. } => panic!("expected entry must be found"),
        };
    assert_eq!(
        rank_of(&first),
        rank_of(&second),
        "two replays of the same snapshot and cases must agree exactly"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn replay_refuses_a_snapshot_taken_under_a_different_embedding_model() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-1", at_angle(0.0), MODEL_A).await;
    let manifest = take_snapshot(&fx.pool, USER, "snap-mismatch")
        .await
        .expect("take_snapshot succeeds");

    // A case with no cached embedding under the snapshot's own model at
    // all — nothing here can produce the vector this replay needs.
    add_case(
        &fx.pool,
        USER,
        CaseInput {
            query_text: "a query embedded under a retired model",
            expected_entry_id: "kb-1",
            source_request_id: None,
            note: Some("no cache under the snapshot's model, on purpose"),
            baseline_snapshot_id: None,
        },
    )
    .await
    .expect("add_case succeeds");

    let outcome = run_replay(&fx.pool, USER, &manifest).await;
    let err = outcome.expect_err("replay must refuse without a cached embedding under the model");
    let message = err.to_string();
    assert!(
        message.contains(MODEL_A),
        "the refusal must name the snapshot's own embedding model: {message}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_case_whose_expected_entry_no_longer_exists_is_reported_not_skipped() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-1", at_angle(0.0), MODEL_A).await;
    let manifest = take_snapshot(&fx.pool, USER, "snap-missing-entry")
        .await
        .expect("take_snapshot succeeds");
    seed_case(&fx.pool, "kb-never-existed", at_angle(0.1), MODEL_A).await;

    let report = run_replay(&fx.pool, USER, &manifest)
        .await
        .expect("replay succeeds even when a case's expected entry is missing");

    assert_eq!(report.case_count, 1);
    assert_eq!(
        report.results.len(),
        1,
        "the case must be reported, not skipped"
    );
    match &report.results[0].outcome {
        CaseOutcome::ExpectedEntryMissing { .. } => {}
        CaseOutcome::Ranked { .. } => panic!("an entry that was never seeded cannot have ranked"),
    }

    fx.cleanup().await;
}

/// Acceptance (#893, #1341): the live `[Recall]` block never shows an entry
/// whose disposition is anything but `active` or `refuted` -- `recall_admits`
/// in `crates/daemon/src/recall.rs` filters before ranking. Replay must hold
/// the same admission, because `ActivationWeights::disposition` prices every
/// other disposition at a zero penalty on the assumption that the caller has
/// already excluded them; a replay that ranked one anyway would report a
/// rank production could never produce.
#[tokio::test]
async fn replay_never_ranks_a_candidate_the_live_block_would_not_admit() {
    let Some(fx) = fixture().await else { return };

    // The distractor sits at the query's own angle -- distance 0, the best
    // any row can score -- so if disposition admission were skipped it would
    // rank first and bury the expected entry.
    seed_entry(&fx.pool, "kb-expected", at_angle(0.3), MODEL_A).await;
    seed_entry(&fx.pool, "kb-distractor", at_angle(0.0), MODEL_A).await;
    seed_disposition(&fx.pool, "kb-distractor", "obsolete").await;

    let manifest = take_snapshot(&fx.pool, USER, "snap-disposition")
        .await
        .expect("take_snapshot succeeds");
    seed_case(&fx.pool, "kb-expected", at_angle(0.0), MODEL_A).await;

    let report = run_replay(&fx.pool, USER, &manifest)
        .await
        .expect("replay succeeds");

    assert_eq!(report.results.len(), 1);
    match &report.results[0].outcome {
        CaseOutcome::Ranked { rank, top, .. } => {
            assert_eq!(
                *rank, 1,
                "the obsolete distractor must never outrank the expected entry, however \
                 close its distance"
            );
            assert!(
                !top.iter().any(|c| c.entry_id == "kb-distractor"),
                "an obsolete entry must never appear among the ranked candidates at all: {top:?}"
            );
        }
        CaseOutcome::ExpectedEntryMissing { .. } => panic!("kb-expected was seeded and admitted"),
    }

    fx.cleanup().await;
}

#[tokio::test]
async fn the_replay_report_states_the_case_count_and_never_truncates_silently() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-1", at_angle(0.0), MODEL_A).await;
    seed_entry(&fx.pool, "kb-2", at_angle(0.3), MODEL_A).await;
    let manifest = take_snapshot(&fx.pool, USER, "snap-count")
        .await
        .expect("take_snapshot succeeds");

    for i in 0..3 {
        seed_case(&fx.pool, "kb-1", at_angle(0.05 * i as f32), MODEL_A).await;
    }

    let report = run_replay(&fx.pool, USER, &manifest)
        .await
        .expect("replay succeeds");

    assert_eq!(report.case_count, 3, "the case count is stated");
    assert_eq!(
        report.results.len(),
        report.case_count,
        "the report must never truncate the case list"
    );
    assert_eq!(
        report.scorer_version,
        desktop_assistant_core::domain::activation::ACTIVATION_SCORER_VERSION,
        "the report carries the scorer version it actually ranked under, so two reports \
         ranked by different builds are never mistaken for the same experiment"
    );
    assert!(
        report.too_small_to_generalize,
        "three cases is well under the small-set threshold"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn replay_ranks_a_seeded_corpus_exactly_as_the_live_scan_ranks_it() {
    let Some(fx) = fixture().await else { return };

    // Query sits outside the cluster rather than in its middle, so every
    // entry's distance from it is distinct — a tie would let the two scans
    // agree on the *set* of nearest rows while disagreeing on tie order for
    // reasons that have nothing to do with distance parity.
    //
    // kb-0 carries the query's own direction but at five times its
    // magnitude. Cosine distance ignores magnitude entirely, so kb-0 stays
    // nearest; Euclidean distance does not, and ranks kb-0 last. Unit
    // vectors alone cannot tell the two operators apart — cosine distance
    // and squared Euclidean distance are a monotonic transform of each
    // other on the unit sphere — so this row is what makes the assertion
    // below actually pin the operator rather than merely the angle.
    let query = at_angle(0.0);
    seed_entry(&fx.pool, "kb-0", vec![5.0, 0.0, 0.0], MODEL_A).await;
    let ids = ["kb-1", "kb-2", "kb-3", "kb-4"];
    let angles = [0.15_f32, 0.4, 0.9, 1.5];
    for (id, angle) in ids.iter().zip(angles.iter()) {
        seed_entry(&fx.pool, id, at_angle(*angle), MODEL_A).await;
    }

    let manifest = take_snapshot(&fx.pool, USER, "snap-parity")
        .await
        .expect("take_snapshot succeeds");

    let live_store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    let live_order: Vec<String> = with_user_id(UserId::new(USER), async {
        live_store
            .nearest_by_embedding(query.clone(), MODEL_A, 10)
            .await
            .expect("live scan succeeds")
            .entries
            .into_iter()
            .map(|(entry, _distance)| entry.id)
            .collect()
    })
    .await;

    let snapshot_order: Vec<String> =
        ranked_snapshot_scan(&fx.pool, USER, &manifest.id, query, Utc::now())
            .await
            .expect("snapshot scan succeeds")
            .into_iter()
            .map(|r| r.entry_id)
            .collect();

    assert_eq!(
        live_order, snapshot_order,
        "the frozen scan must order the same corpus exactly as the live scan does"
    );
    assert_eq!(
        live_order.len(),
        ids.len() + 1,
        "kb-0 plus the four angled entries"
    );
    assert_eq!(
        live_order.first().map(String::as_str),
        Some("kb-0"),
        "cosine distance ignores magnitude, so kb-0 (same direction, larger magnitude) is \
         nearest under the correct operator"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_case_without_a_source_or_a_note_is_refused() {
    let Some(fx) = fixture().await else { return };

    let result = add_case(
        &fx.pool,
        USER,
        CaseInput {
            query_text: "an untraceable query",
            expected_entry_id: "kb-1",
            source_request_id: None,
            note: None,
            baseline_snapshot_id: None,
        },
    )
    .await;

    assert!(
        result.is_err(),
        "a case with neither a source turn nor a note must be refused"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn a_case_with_a_source_is_accepted() {
    let Some(fx) = fixture().await else { return };

    let result = add_case(
        &fx.pool,
        USER,
        CaseInput {
            query_text: "a traceable query",
            expected_entry_id: "kb-1",
            source_request_id: Some("req-123"),
            note: None,
            baseline_snapshot_id: None,
        },
    )
    .await;

    assert!(
        result.is_ok(),
        "a case naming its source turn must be accepted: {:?}",
        result.err()
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn snapshot_drop_refuses_while_a_case_baseline_names_it() {
    let Some(fx) = fixture().await else { return };

    seed_entry(&fx.pool, "kb-1", at_angle(0.0), MODEL_A).await;
    let baselined = take_snapshot(&fx.pool, USER, "snap-baselined")
        .await
        .expect("take_snapshot succeeds");
    let unbaselined = take_snapshot(&fx.pool, USER, "snap-unbaselined")
        .await
        .expect("take_snapshot succeeds");

    let case_id = seed_case(&fx.pool, "kb-1", at_angle(0.1), MODEL_A).await;
    set_case_baseline(&fx.pool, USER, &case_id, &baselined.id)
        .await
        .expect("set_case_baseline succeeds");

    let refused = drop_snapshot(&fx.pool, USER, &baselined.id).await;
    assert!(
        refused.is_err(),
        "a snapshot a case's baseline still names must not be dropped"
    );

    // Paired with a permit: an unreferenced snapshot drops cleanly, so the
    // refusal above is the baseline link and not some other cause.
    let permitted = drop_snapshot(&fx.pool, USER, &unbaselined.id).await;
    assert!(
        permitted.is_ok(),
        "a snapshot no case references must drop cleanly: {:?}",
        permitted.err()
    );

    fx.cleanup().await;
}
