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

use desktop_assistant_storage::embedded_tables::{EMBEDDED_TABLE_EXEMPTIONS, EMBEDDED_TABLES};
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
    // A vector table the sweep must never touch (#1328) is not "not yet
    // registered" -- it is deliberately outside EMBEDDED_TABLES, and belongs
    // here with its reason instead. Folded into the same accounted-for set
    // so this test still fails for a vector table in neither list.
    let exempt: BTreeSet<String> = EMBEDDED_TABLE_EXEMPTIONS
        .iter()
        .map(|(t, _reason)| t.to_string())
        .collect();
    let accounted_for: BTreeSet<String> = declared.union(&exempt).cloned().collect();

    assert!(
        !found.is_empty(),
        "the migrations must create at least one embedded table for this test to mean anything"
    );
    let missing: Vec<&String> = found.difference(&accounted_for).collect();
    assert!(
        missing.is_empty(),
        "these tables hold embeddings but are absent from both EMBEDDED_TABLES and \
         EMBEDDED_TABLE_EXEMPTIONS, so nobody has decided whether the lifecycle sweeps should \
         reach them: {missing:?}"
    );
    let phantom: Vec<&String> = declared.difference(&found).collect();
    assert!(
        phantom.is_empty(),
        "EMBEDDED_TABLES names tables that do not hold embeddings: {phantom:?}"
    );
    let phantom_exempt: Vec<&String> = exempt.difference(&found).collect();
    assert!(
        phantom_exempt.is_empty(),
        "EMBEDDED_TABLE_EXEMPTIONS names tables that do not hold embeddings: {phantom_exempt:?}"
    );

    fx.cleanup().await;
}

/// Acceptance (#1328): the exemption is honoured in practice, not only
/// declared. A snapshot's frozen vectors, and a case's per-model cached
/// vector, must survive a model change untouched -- exactly the sweep that
/// would otherwise strand or destroy them.
#[tokio::test]
async fn a_stale_sweep_leaves_exempted_tables_untouched() {
    let Some(fx) = fixture("reg1328exempt").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };

    let snapshot_id = "snap-exempt";
    let case_id = "case-exempt";
    sqlx::query(
        "INSERT INTO recall_snapshots \
             (id, user_id, name, embedding_model, entry_count, use_count, excluded_count) \
         VALUES ($1, 'exempt-user', 'snap', $2, 1, 0, 0)",
    )
    .bind(snapshot_id)
    .bind(old_model())
    .execute(&fx.pool)
    .await
    .expect("seed snapshot manifest");

    let snapshot_vectors: Vec<Vector> = vec![Vector::from(vec![0.1_f32, 0.2, 0.3])];
    sqlx::query(
        "INSERT INTO recall_snapshot_entries \
             (snapshot_id, user_id, entry_id, content, embedding, embedding_model, \
              created_at, updated_at, disposition) \
         VALUES ($1, 'exempt-user', 'kb-frozen', 'frozen content', $2::vector[], $3, \
                 NOW(), NOW(), 'active')",
    )
    .bind(snapshot_id)
    .bind(&snapshot_vectors)
    .bind(old_model())
    .execute(&fx.pool)
    .await
    .expect("seed a frozen snapshot entry");

    sqlx::query(
        "INSERT INTO recall_cases (id, user_id, query_text, expected_entry_id, note) \
         VALUES ($1, 'exempt-user', 'a query', 'kb-frozen', 'seeded for the exemption test')",
    )
    .bind(case_id)
    .execute(&fx.pool)
    .await
    .expect("seed a case");

    let case_vector = Vector::from(vec![0.4_f32, 0.5, 0.6]);
    sqlx::query(
        "INSERT INTO recall_case_embeddings (case_id, user_id, embedding_model, embedding) \
         VALUES ($1, 'exempt-user', $2, $3)",
    )
    .bind(case_id)
    .bind(old_model())
    .bind(&case_vector)
    .execute(&fx.pool)
    .await
    .expect("seed a cached case embedding");

    invalidate_stale_embeddings(&fx.pool, &new_model())
        .await
        .expect("the sweep must complete even though it never visits these tables");

    let (snapshot_has_embedding, snapshot_model): (bool, Option<String>) = sqlx::query_as(
        "SELECT embedding IS NOT NULL, embedding_model FROM recall_snapshot_entries \
         WHERE snapshot_id = $1 AND entry_id = 'kb-frozen'",
    )
    .bind(snapshot_id)
    .fetch_one(&fx.pool)
    .await
    .expect("read the frozen entry back");
    assert!(
        snapshot_has_embedding,
        "a stale sweep must never clear a frozen snapshot's vectors -- there is no backfill \
         that could ever refill them"
    );
    assert_eq!(
        snapshot_model.as_deref(),
        Some(old_model().as_str()),
        "the snapshot's own embedding model must survive a live model change unchanged"
    );

    let case_row_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM recall_case_embeddings WHERE case_id = $1 AND embedding_model = $2",
    )
    .bind(case_id)
    .bind(old_model())
    .fetch_one(&fx.pool)
    .await
    .expect("count the case's cached embeddings");
    assert_eq!(
        case_row_count, 1,
        "a case's old-model cache entry must survive a live model change -- deleting it would \
         strand any snapshot still frozen under that model with no way to ever replay it again"
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

/// Acceptance: a tag the embedder cannot handle loses its stale vector rather
/// than having it relabelled as current. Relabelling would declare a
/// wrong-dimension vector fresh and put it beyond the stale sweep, which only
/// looks at mismatched stamps -- turning a transient embed failure into a
/// permanent search error.
#[tokio::test]
async fn tag_backfill_failure_clears_the_vector_rather_than_relabelling_it() {
    let Some(fx) = fixture("reg682i").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    // Stale stamp, vector still present -- the state a hot model swap leaves
    // behind before any sweep has run.
    embedded_tag(&fx.pool, "billing", &old_model(), 3).await;

    let failing: BackfillEmbedFn =
        Box::new(|_| Box::pin(async { Err("embedder unavailable".to_string()) }));
    backfill_tag_embeddings(&fx.pool, &failing, &new_model())
        .await
        .expect("backfill reports success even when individual rows fail");

    let (has_embedding, model) = tag_state(&fx.pool, "billing").await;
    assert!(
        !has_embedding,
        "a failed embed must clear the superseded vector, not keep it"
    );
    assert_eq!(
        model.as_deref(),
        Some(new_model().as_str()),
        "the row must be stamped attempted so it is not retried in a tight loop"
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
    let embedded = backfill_tag_embeddings(
        &fx.pool,
        &recording_embed(Arc::clone(&seen), 4),
        &new_model(),
    )
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
    backfill_tag_embeddings(
        &fx.pool,
        &recording_embed(Arc::clone(&seen), 4),
        &new_model(),
    )
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

/// Acceptance: a tag registered without a description embeds as its name alone,
/// on the backfill path as well as the creation path.
///
/// A tag written through the knowledge-base tool can arrive with no
/// description, because a write must never fail over a missing one. The two
/// paths must agree on what that tag's embed text is, or its backfilled vector
/// stops being comparable with the vectors new tags are matched against.
#[tokio::test]
async fn tag_backfill_reproduces_the_creation_embed_text_without_a_description() {
    let Some(fx) = fixture("reg1070a").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    sqlx::query(
        "INSERT INTO tag_registry (user_id, name, description, examples, distinguish_from) \
         VALUES ($1, 'topic:deploy', '', '[]'::jsonb, '{}')",
    )
    .bind(USER)
    .execute(&fx.pool)
    .await
    .expect("seed a described-less tag");

    let seen = Arc::new(Mutex::new(Vec::new()));
    backfill_tag_embeddings(
        &fx.pool,
        &recording_embed(Arc::clone(&seen), 4),
        &new_model(),
    )
    .await
    .expect("backfill tags");

    let texts = seen.lock().expect("read recorded texts").clone();
    assert_eq!(
        texts,
        vec!["topic:deploy".to_string()],
        "with no description the embed text is the name alone, with no trailing \
         separator, on both paths"
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

/// Acceptance: an embedder that returns fewer vectors than it was given texts
/// must not strand the unmatched rows. They still match the backfill's own
/// SELECT, so leaving them unwritten would re-select them on every pass -- an
/// unbounded loop that bills a metered provider for each attempt.
#[tokio::test]
async fn tag_backfill_short_batch_does_not_strand_rows() {
    let Some(fx) = fixture("reg682j").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    for name in ["billing", "invoicing", "payroll"] {
        sqlx::query(
            "INSERT INTO tag_registry (user_id, name, description, examples, distinguish_from) \
             VALUES ($1, $2, 'seeded', '[]'::jsonb, '{}')",
        )
        .bind(USER)
        .bind(name)
        .execute(&fx.pool)
        .await
        .expect("seed tag");
    }

    // Honours the port's contract for a single text, violates it for a batch --
    // the shape a provider that silently caps batch size produces.
    let short: BackfillEmbedFn = Box::new(|texts: Vec<String>| {
        Box::pin(async move {
            if texts.len() == 1 {
                Ok(vec![vec![0.5_f32; 4]])
            } else {
                Ok(Vec::new())
            }
        })
    });

    let embedded = backfill_tag_embeddings(&fx.pool, &short, &new_model())
        .await
        .expect("backfill must return rather than loop");

    assert_eq!(
        embedded, 3,
        "every row must be accounted for, via the retry path"
    );
    for name in ["billing", "invoicing", "payroll"] {
        let (has_embedding, model) = tag_state(&fx.pool, name).await;
        assert!(
            has_embedding,
            "{name} must be embedded by the per-row retry"
        );
        assert_eq!(model.as_deref(), Some(new_model().as_str()));
    }

    fx.cleanup().await;
}

/// Acceptance: the backfill writes only within one tenant. `name` alone looks
/// like a key and was the primary key before migration 016, so the `user_id`
/// half of the predicate is the only thing stopping a maintenance sweep from
/// overwriting another tenant's vector with text embedded from this one's.
#[tokio::test]
async fn tag_backfill_scopes_writes_per_user() {
    let Some(fx) = fixture("reg682k").await else {
        eprintln!("skip: TEST_DATABASE_URL not set");
        return;
    };
    for (user, description) in [("alice", "alice invoices"), ("bob", "bob timesheets")] {
        sqlx::query(
            "INSERT INTO tag_registry (user_id, name, description, examples, distinguish_from) \
             VALUES ($1, 'billing', $2, '[]'::jsonb, '{}')",
        )
        .bind(user)
        .bind(description)
        .execute(&fx.pool)
        .await
        .expect("seed tag");
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    backfill_tag_embeddings(
        &fx.pool,
        &recording_embed(Arc::clone(&seen), 4),
        &new_model(),
    )
    .await
    .expect("backfill tags");

    let mut texts = seen.lock().expect("read recorded texts").clone();
    texts.sort();
    assert_eq!(
        texts,
        vec![
            "billing: alice invoices".to_string(),
            "billing: bob timesheets".to_string()
        ],
        "each tenant's row must be embedded from its own description"
    );

    let rows: Vec<(String, Option<String>)> =
        sqlx::query_as("SELECT user_id, embedding_model FROM tag_registry ORDER BY user_id")
            .fetch_all(&fx.pool)
            .await
            .expect("probe both tenants");
    assert_eq!(rows.len(), 2, "both rows must survive the sweep");
    for (user, model) in rows {
        assert_eq!(
            model.as_deref(),
            Some(new_model().as_str()),
            "{user}'s row must be stamped"
        );
    }

    fx.cleanup().await;
}
