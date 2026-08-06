//! The two reads behind the `[Recall]` block (issue #1100).
//!
//! `PgKnowledgeBaseStore::nearest_by_embedding` and
//! `tag_registry::nearest_tags` are what a turn asks before the model's first
//! move. Both are new query surfaces over personal data, so the suite pins the
//! three properties the block's correctness rests on.
//!
//! 1. **One user's rows and no other's.** Row-level security is a non-FORCE
//!    backstop the table owner bypasses, so the `WHERE user_id` predicate in
//!    each query is the only real guard. A read that leaked another tenant's
//!    memory would put it in front of the model as this user's own.
//! 2. **Nearest first, with a usable distance.** The block sets a relevance
//!    floor over the distance these queries return. An unordered result, or one
//!    whose distance is not a cosine distance, would make the floor meaningless.
//! 3. **Only rows embedded by the query's own model.** A stored vector from
//!    another model has another dimension, and the vector operator answers that
//!    with an error rather than a miss - which would fail the read instead of
//!    degrading it.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test recall_candidates
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_storage::tag_registry::nearest_tags;
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgPool, UserId, with_user_id};
use pgvector::Vector;

const USER: &str = "recall-user";
const OTHER_USER: &str = "recall-other-user";

/// The model every seeded vector is stamped with, and the one every read below
/// passes.
const MODEL: &str = "recall-test-model";

/// A model of another dimension entirely, used to prove the scope predicate
/// keeps incompatible vectors out of the comparison.
const OTHER_MODEL: &str = "recall-other-model";

async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("recall1100").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

/// A three-dimensional unit vector pointing along one axis. Cosine distance
/// between two of these is 1.0; between a vector and itself it is 0.0, which
/// makes every expectation below readable without a tolerance argument.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 3];
    v[i] = 1.0;
    v
}

/// Write an entry as `user`, then stamp an embedding on it. Writes never embed
/// inline - the background backfill does - so the test stands in for it.
async fn seed_entry(pool: &PgPool, user: &str, id: &str, content: &str, chunk: Vec<f32>) {
    seed_entry_with_model(pool, user, id, content, chunk, MODEL).await;
}

async fn seed_entry_with_model(
    pool: &PgPool,
    user: &str,
    id: &str,
    content: &str,
    chunk: Vec<f32>,
    model: &str,
) {
    let store = PgKnowledgeBaseStore::new(pool.clone());
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(id, content, vec!["topic".to_string()]))
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

/// Insert a registry row directly so the test owns its vector and its model
/// stamp, rather than going through the dedup path that would compute both.
async fn seed_tag(pool: &PgPool, user: &str, name: &str, chunk: Vec<f32>, model: &str) {
    let vector = Vector::from(chunk);
    sqlx::query(
        "INSERT INTO tag_registry \
            (user_id, name, description, examples, distinguish_from, embedding, embedding_model) \
         VALUES ($1, $2, 'seeded', '[]'::jsonb, '{}', $3, $4)",
    )
    .bind(user)
    .bind(name)
    .bind(&vector)
    .bind(model)
    .execute(pool)
    .await
    .expect("seed tag");
}

// -- the knowledge arm -------------------------------------------------------

#[tokio::test]
async fn nearest_entries_come_back_nearest_first_with_their_distance() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    seed_entry(&fx.pool, USER, "kb-near", "the near one", axis(0)).await;
    seed_entry(&fx.pool, USER, "kb-far", "the far one", axis(1)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0.id, "kb-near", "nearest first");
        assert!(
            hits[0].1 < 1e-6,
            "a vector against itself is at distance 0, got {}",
            hits[0].1
        );
        assert_eq!(hits[1].0.id, "kb-far");
        assert!(
            (hits[1].1 - 1.0).abs() < 1e-6,
            "orthogonal vectors are at cosine distance 1, got {}",
            hits[1].1
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_never_cross_the_user_boundary() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // The other tenant's entry is a perfect match for the query vector, so
    // anything but an explicit scope would rank it first.
    seed_entry(&fx.pool, OTHER_USER, "kb-theirs", "their secret", axis(0)).await;
    seed_entry(&fx.pool, USER, "kb-mine", "my own note", axis(1)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let ids: Vec<&str> = hits.iter().map(|(e, _)| e.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["kb-mine"],
            "recall must never offer another tenant's memory as this user's own"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_ignore_a_row_embedded_by_another_model() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    // Four dimensions against the query's three: comparing them raises rather
    // than missing, so the scope predicate must keep the row out entirely.
    seed_entry_with_model(
        &fx.pool,
        USER,
        "kb-other-model",
        "embedded by something else",
        vec![1.0, 0.0, 0.0, 0.0],
        OTHER_MODEL,
    )
    .await;
    seed_entry(&fx.pool, USER, "kb-mine", "my own note", axis(0)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("a row from another model must not fail the read");

        let ids: Vec<&str> = hits.iter().map(|(e, _)| e.id.as_str()).collect();
        assert_eq!(ids, vec!["kb-mine"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_stop_at_the_scan_limit() {
    // The limit is what makes the block's "and N more" a bounded count rather
    // than an unbounded read.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    for i in 0..5 {
        seed_entry(&fx.pool, USER, &format!("kb-{i}"), "a note", axis(0)).await;
    }

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 3)
            .await
            .expect("the read succeeds");
        assert_eq!(hits.len(), 3);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_skip_a_retired_entry() {
    // A soft-deleted entry is hidden from every other read path, and offering
    // it back as a recall candidate would resurrect it in the model's view.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    seed_entry(&fx.pool, USER, "kb-retired", "a retired note", axis(0)).await;
    sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind("kb-retired")
        .execute(&fx.pool)
        .await
        .expect("retire the entry");

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");
        assert!(hits.is_empty(), "a retired entry is not a candidate");
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_entries_answer_nothing_without_an_embedding() {
    // No embedding means no vector arm. The caller has a full-text path to fall
    // back to; what it must not get is an error from a zero-dimension vector.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    seed_entry(&fx.pool, USER, "kb-mine", "my own note", axis(0)).await;

    with_user_id(UserId::new(USER), async {
        let hits = store
            .nearest_by_embedding(Vec::new(), MODEL, 10)
            .await
            .expect("an absent embedding is not an error");
        assert!(hits.is_empty());
    })
    .await;

    fx.cleanup().await;
}

// -- the tag arm -------------------------------------------------------------

#[tokio::test]
async fn nearest_tags_come_back_nearest_first_with_their_distance() {
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, USER, "topic:near", axis(0), MODEL).await;
    seed_tag(&fx.pool, USER, "topic:far", axis(1), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["topic:near", "topic:far"]);
        assert!(hits[0].1 < 1e-6, "got {}", hits[0].1);
        assert!((hits[1].1 - 1.0).abs() < 1e-6, "got {}", hits[1].1);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_never_cross_the_user_boundary() {
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, OTHER_USER, "topic:theirs", axis(0), MODEL).await;
    seed_tag(&fx.pool, USER, "topic:mine", axis(1), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["topic:mine"],
            "another tenant's vocabulary is not this user's"
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_skip_a_deprecated_tag() {
    // A deprecated tag points at its replacement. Offering it as vocabulary
    // would send the model's next search at a name no row carries any more.
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, USER, "topic:current", axis(1), MODEL).await;
    seed_tag(&fx.pool, USER, "topic:retired", axis(0), MODEL).await;
    sqlx::query("UPDATE tag_registry SET deprecated_for_tag = $1 WHERE user_id = $2 AND name = $3")
        .bind("topic:current")
        .bind(USER)
        .bind("topic:retired")
        .execute(&fx.pool)
        .await
        .expect("deprecate the tag");

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("the read succeeds");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["topic:current"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_ignore_a_row_embedded_by_another_model() {
    let Some(fx) = fixture().await else { return };

    seed_tag(
        &fx.pool,
        USER,
        "topic:other-model",
        vec![1.0, 0.0, 0.0, 0.0],
        OTHER_MODEL,
    )
    .await;
    seed_tag(&fx.pool, USER, "topic:mine", axis(0), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, axis(0), MODEL, 10)
            .await
            .expect("a row from another model must not fail the read");

        let names: Vec<&str> = hits.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["topic:mine"]);
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn nearest_tags_answer_nothing_without_an_embedding() {
    // The registry carries no full-text index, so the tag arm has nothing to
    // degrade to. It goes quiet rather than raising.
    let Some(fx) = fixture().await else { return };

    seed_tag(&fx.pool, USER, "topic:mine", axis(0), MODEL).await;

    with_user_id(UserId::new(USER), async {
        let hits = nearest_tags(&fx.pool, Vec::new(), MODEL, 10)
            .await
            .expect("an absent embedding is not an error");
        assert!(hits.is_empty());
    })
    .await;

    fx.cleanup().await;
}
