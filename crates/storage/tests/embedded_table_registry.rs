//! Every table holding embeddings must participate in the embedding lifecycle
//! (issue #682).
//!
//! `invalidate_stale_embeddings` used to sweep a hand-written list of two
//! tables, `knowledge_base` and `tool_definitions`. `skill_index` and
//! `tag_registry` were added later and silently opted out, so a model change
//! left their vectors at the old dimension while queries embedded at the new
//! one -- and pgvector raises `different vector dimensions` rather than
//! degrading. `tag_registry` had no backfill either, so nothing converged it.
//!
//! These tests pin the coverage itself, not just the two tables that happened
//! to be covered: the schema is the source of truth, so a future table with a
//! `vector` column fails the registry test until it is declared.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test embedded_table_registry
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Mutex;

use desktop_assistant_storage::embedded_tables::EMBEDDED_TABLES;
use desktop_assistant_storage::embedding_backfill::{
    BackfillEmbedFn, backfill_tag_embeddings, invalidate_stale_embeddings,
};
use desktop_assistant_storage::tag_registry::{TagProposal, create_or_match_tag};
use desktop_assistant_storage::{UserId, run_migrations, with_user_id};
use pgvector::Vector;
use sqlx::PgPool;

const USER: &str = "tag-owner";
const DIGEST_OLD: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const DIGEST_NEW: &str = "2222222222222222222222222222222222222222222222222222222222222222";

fn old_model() -> String {
    format!("nomic-embed-text@{DIGEST_OLD}")
}

fn new_model() -> String {
    format!("amazon.titan-embed-text-v2:0@{DIGEST_NEW}")
}

async fn fixture(prefix: &str) -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new(prefix).await?;
    run_migrations(&fx.pool).await.expect("run_migrations");
    Some(fx)
}

/// Seed one tag row already embedded under `model`, with `dims` dimensions so a
/// dimension mismatch can be provoked deliberately.
async fn embedded_tag(pool: &PgPool, name: &str, model: &str, dims: usize) {
    let vector = Vector::from(vec![0.1_f32; dims]);
    sqlx::query(
        "INSERT INTO tag_registry \
            (user_id, name, description, examples, distinguish_from, embedding, embedding_model) \
         VALUES ($1, $2, $3, '[]'::jsonb, '{}', $4, $5)",
    )
    .bind(USER)
    .bind(name)
    .bind("a seeded tag")
    .bind(&vector)
    .bind(model)
    .execute(pool)
    .await
    .expect("seed tag row");
}

/// Seed one skill row already embedded under `model`.
async fn embedded_skill(pool: &PgPool, name: &str, model: &str) {
    let vecs: Vec<Vector> = vec![Vector::from(vec![0.1_f32, 0.2, 0.3])];
    sqlx::query(
        "INSERT INTO skill_index \
            (name, description, disk_path, content_hash, embedding, embedding_model) \
         VALUES ($1, $2, $3, $4, $5::vector[], $6)",
    )
    .bind(name)
    .bind("a seeded skill")
    .bind("/tmp/skills/seeded")
    .bind("deadbeef")
    .bind(&vecs)
    .bind(model)
    .execute(pool)
    .await
    .expect("seed skill row");
}

async fn tag_state(pool: &PgPool, name: &str) -> (bool, Option<String>) {
    sqlx::query_as::<_, (bool, Option<String>)>(
        "SELECT embedding IS NOT NULL, embedding_model FROM tag_registry WHERE name = $1",
    )
    .bind(name)
    .fetch_one(pool)
    .await
    .expect("probe tag")
}

async fn skill_state(pool: &PgPool, name: &str) -> (bool, Option<String>) {
    sqlx::query_as::<_, (bool, Option<String>)>(
        "SELECT embedding IS NOT NULL, embedding_model FROM skill_index WHERE name = $1",
    )
    .bind(name)
    .fetch_one(pool)
    .await
    .expect("probe skill")
}

/// An embed function that records the texts it was handed and returns
/// `dims`-dimensional vectors.
fn recording_embed(seen: Arc<Mutex<Vec<String>>>, dims: usize) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        let seen = Arc::clone(&seen);
        Box::pin(async move {
            let n = texts.len();
            seen.lock().expect("record texts").extend(texts);
            Ok(vec![vec![0.5_f32; dims]; n])
        })
    })
}

/// Acceptance: the sweep's coverage is declared, and the schema proves it
/// complete. This is the test that would have caught both missing tables.
#[tokio::test]
async fn every_table_with_a_vector_column_is_registered() {
    let Some(fx) = fixture("reg682a").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    let in_schema: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT table_name FROM information_schema.columns \
         WHERE table_schema = $1 AND udt_name IN ('vector', '_vector')",
    )
    .bind(fx.schema())
    .fetch_all(&fx.pool)
    .await
    .expect("enumerate vector columns");

    let found: BTreeSet<String> = in_schema.into_iter().map(|(t,)| t).collect();
    let declared: BTreeSet<String> = EMBEDDED_TABLES.iter().map(|t| t.to_string()).collect();

    assert!(
        !found.is_empty(),
        "the migrations must create at least one embedded table for this test to mean anything"
    );
    let missing: Vec<&String> = found.difference(&declared).collect();
    assert!(
        missing.is_empty(),
        "these tables hold embeddings but are absent from EMBEDDED_TABLES, so the \
         lifecycle sweeps skip them and a model change strands their vectors: {missing:?}"
    );
    let phantom: Vec<&String> = declared.difference(&found).collect();
    assert!(
        phantom.is_empty(),
        "EMBEDDED_TABLES names tables that do not hold embeddings: {phantom:?}"
    );

    fx.cleanup().await;
}

/// Acceptance: a superseded model stamp invalidates tag vectors, closing the
/// dimension-mismatch window before any search runs.
#[tokio::test]
async fn stale_sweep_invalidates_tag_registry_embeddings() {
    let Some(fx) = fixture("reg682b").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    embedded_tag(&fx.pool, "billing", &old_model(), 3).await;

    invalidate_stale_embeddings(&fx.pool, &new_model())
        .await
        .expect("invalidate");

    let (has_embedding, model) = tag_state(&fx.pool, "billing").await;
    assert!(
        !has_embedding,
        "a tag embedded under a superseded model must be invalidated, or the next \
         nearest-neighbour search compares mismatched dimensions and errors"
    );
    assert!(
        model.is_none(),
        "the stale stamp must clear with the vector so the backfill re-embeds it"
    );

    fx.cleanup().await;
}

/// Acceptance: the same for skills.
#[tokio::test]
async fn stale_sweep_invalidates_skill_index_embeddings() {
    let Some(fx) = fixture("reg682c").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    embedded_skill(&fx.pool, "deploy-runbook", &old_model()).await;

    invalidate_stale_embeddings(&fx.pool, &new_model())
        .await
        .expect("invalidate");

    let (has_embedding, model) = skill_state(&fx.pool, "deploy-runbook").await;
    assert!(
        !has_embedding,
        "a skill embedded under a superseded model must be invalidated; the backfill \
         converges it eventually, but searches in that window error on dimension mismatch"
    );
    assert!(model.is_none());

    fx.cleanup().await;
}

/// Acceptance: tombstones are swept too. A soft-deleted row is invisible to
/// search, so its stale vector looks harmless -- until the row is restored from
/// the trash carrying a vector of the wrong dimension.
#[tokio::test]
async fn soft_deleted_knowledge_rows_are_swept_so_restore_cannot_resurrect_a_stale_vector() {
    let Some(fx) = fixture("reg682h").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    let vecs: Vec<Vector> = vec![Vector::from(vec![0.1_f32, 0.2, 0.3])];
    sqlx::query(
        "INSERT INTO knowledge_base \
            (id, user_id, content, tags, embedding, embedding_model, deleted_at) \
         VALUES ($1, $2, $3, '{}', $4::vector[], $5, NOW())",
    )
    .bind("tombstone")
    .bind(USER)
    .bind("deleted but restorable")
    .bind(&vecs)
    .bind(old_model())
    .execute(&fx.pool)
    .await
    .expect("seed soft-deleted row");

    invalidate_stale_embeddings(&fx.pool, &new_model())
        .await
        .expect("invalidate");

    let (has_embedding,): (bool,) =
        sqlx::query_as("SELECT embedding IS NOT NULL FROM knowledge_base WHERE id = $1")
            .bind("tombstone")
            .fetch_one(&fx.pool)
            .await
            .expect("probe tombstone");
    assert!(
        !has_embedding,
        "a soft-deleted row keeps its vector today; restoring it after a model change \
         puts a wrong-dimension vector back into search"
    );

    fx.cleanup().await;
}

/// Acceptance: the digest-restamp path (#655) extends to the new tables -- a
/// cosmetic rename with an unchanged digest must not discard vectors.
#[tokio::test]
async fn matching_digest_restamps_tag_instead_of_invalidating() {
    let Some(fx) = fixture("reg682d").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    embedded_tag(
        &fx.pool,
        "billing",
        &format!("nomic-embed-text:latest@{DIGEST_OLD}"),
        3,
    )
    .await;
    let current = old_model();

    invalidate_stale_embeddings(&fx.pool, &current)
        .await
        .expect("invalidate");

    let (has_embedding, model) = tag_state(&fx.pool, "billing").await;
    assert!(
        has_embedding,
        "same digest means the same model; a rename must not re-embed the corpus"
    );
    assert_eq!(
        model.as_deref(),
        Some(current.as_str()),
        "the row must adopt the new spelling so the comparison converges"
    );

    fx.cleanup().await;
}

/// Acceptance: invalidated tags converge instead of vanishing from dedup.
#[tokio::test]
async fn tag_backfill_re_embeds_invalidated_rows() {
    let Some(fx) = fixture("reg682e").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    embedded_tag(&fx.pool, "billing", &old_model(), 3).await;
    invalidate_stale_embeddings(&fx.pool, &new_model())
        .await
        .expect("invalidate");

    let seen = Arc::new(Mutex::new(Vec::new()));
    let embedded = backfill_tag_embeddings(&fx.pool, &recording_embed(Arc::clone(&seen), 4), &new_model())
        .await
        .expect("backfill tags");

    assert_eq!(embedded, 1, "the invalidated tag must be re-embedded");
    let (has_embedding, model) = tag_state(&fx.pool, "billing").await;
    assert!(
        has_embedding,
        "without a tag backfill an invalidated tag is excluded from dedup permanently"
    );
    assert_eq!(model.as_deref(), Some(new_model().as_str()));

    fx.cleanup().await;
}

/// Acceptance: the backfill embeds the same text the creation path does, or the
/// stored vectors are not comparable with the ones new tags are matched against.
#[tokio::test]
async fn tag_backfill_reproduces_the_creation_embed_text() {
    let Some(fx) = fixture("reg682f").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    sqlx::query(
        "INSERT INTO tag_registry (user_id, name, description, examples, distinguish_from) \
         VALUES ($1, 'billing', 'invoices and payments', '[]'::jsonb, '{}')",
    )
    .bind(USER)
    .execute(&fx.pool)
    .await
    .expect("seed unembedded tag");

    let seen = Arc::new(Mutex::new(Vec::new()));
    backfill_tag_embeddings(&fx.pool, &recording_embed(Arc::clone(&seen), 4), &new_model())
        .await
        .expect("backfill tags");

    let texts = seen.lock().expect("read recorded texts").clone();
    assert_eq!(
        texts,
        vec!["billing: invoices and payments".to_string()],
        "create_or_match_tag embeds `name: description`; the backfill must match it"
    );

    fx.cleanup().await;
}

/// Acceptance: the end state -- creating a tag against a registry that held
/// vectors from a superseded model succeeds instead of raising a pgvector
/// dimension error.
#[tokio::test]
async fn tag_creation_survives_a_superseded_model_in_the_registry() {
    let Some(fx) = fixture("reg682g").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    // Three dimensions, stamped with the old model, as prod's 26 rows are.
    embedded_tag(&fx.pool, "billing", &old_model(), 3).await;

    invalidate_stale_embeddings(&fx.pool, &new_model())
        .await
        .expect("invalidate");

    let seen = Arc::new(Mutex::new(Vec::new()));
    // Four dimensions: a different model, as Titan is to nomic.
    let embed = recording_embed(Arc::clone(&seen), 4);
    let outcome = with_user_id(UserId::new(USER), async {
        create_or_match_tag(
            &fx.pool,
            &embed,
            &new_model(),
            TagProposal {
                name: "invoicing".to_string(),
                description: "sending bills".to_string(),
                examples: vec![],
                distinguish_from: vec![],
            },
        )
        .await
    })
    .await;

    assert!(
        outcome.is_ok(),
        "a registry holding superseded-model vectors must not fail tag creation: {:?}",
        outcome.err()
    );

    fx.cleanup().await;
}
