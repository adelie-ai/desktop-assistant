//! Every vector search is scoped to the model that produced its query vector
//! (issue #684).
//!
//! pgvector answers a comparison between vectors of different dimensions with
//! `ERROR: different vector dimensions`, not with a miss. A table that holds
//! rows from two models therefore does not degrade, it breaks -- and a table
//! legitimately holds two models' vectors during any reindex, after a failed
//! stale-embedding sweep, and for the whole of a live embedding-client swap.
//!
//! The predicate on each vector arm makes a row embedded under any other model
//! invisible instead of fatal. Two rules it must respect:
//!
//! * The **full-text arm is never model-scoped**. Filtering it too would make
//!   content unfindable after a model change rather than degrading to lexical
//!   search -- worse than the error being fixed.
//! * **Staleness is decided on the digest** (#655). The stamp is
//!   `<name>@<digest>`; `nomic-embed-text:latest@<d>` and
//!   `nomic-embed-text@<d>` are the same model, and
//!   `invalidate_stale_embeddings` restamps such rows rather than discarding
//!   their vectors. A whole-string predicate would blind the search to every
//!   not-yet-restamped row after a cosmetic rename, so the search compares
//!   digests wherever both sides carry one.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test search_embedding_model_scope
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::{
    IndexedSkill, KnowledgeEntry, Locality, SkillKind, SkillScope, TrustTier,
};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use desktop_assistant_core::skill_catalog::reconcile_scan;
use desktop_assistant_storage::embedding_backfill::BackfillEmbedFn;
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::tag_registry::{CreateTagOutcome, TagProposal, create_or_match_tag};
use desktop_assistant_storage::{PgKnowledgeBaseStore, PgSkillIndexStore, UserId, with_user_id};
use pgvector::Vector;
use sqlx::PgPool;

/// A throwaway schema with migrations applied, or `None` when no database is
/// configured (the suite then pass-skips loudly via `support`).
async fn fixture(name: &str) -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new(name).await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set; {name} pass-skipped");
    }
    fx
}

const USER: &str = "kb-owner";
const DIGEST: &str = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f";
const OTHER_DIGEST: &str = "9f77d1c0b4a2e58631d0c9a7f4b2e8d6c3a1907e5b4d2f8a6c0e9b7d5a3f1c28";

/// The model this daemon is currently embedding with.
fn current() -> String {
    format!("nomic-embed-text@{DIGEST}")
}

/// The same model spelled differently -- a cosmetic config rename that resolves
/// to an identical digest (#655).
fn renamed() -> String {
    format!("nomic-embed-text:latest@{DIGEST}")
}

/// A genuinely different model.
fn superseded() -> String {
    format!("mxbai-embed-large@{OTHER_DIGEST}")
}

/// A query vector under the current model. Three dimensions.
fn query_vec() -> Vec<f32> {
    vec![1.0, 0.0, 0.0]
}

/// A vector under the superseded model. Four dimensions, so any comparison
/// against [`query_vec`] raises rather than missing.
fn other_dimension_vec() -> Vec<f32> {
    vec![1.0, 0.0, 0.0, 0.0]
}

// --- knowledge base ---------------------------------------------------------

async fn write_kb(pool: &PgPool, user: &str, id: &str, content: &str) {
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(id, content, vec!["notes".to_string()]))
            .await
            .expect("kb write");
    })
    .await;
}

/// Stamp a row's vectors and model the way the backfill does. `model = None`
/// leaves the stamp NULL: a vector whose provenance is unknown.
async fn stamp_kb(pool: &PgPool, id: &str, chunk: Vec<f32>, model: Option<&str>) {
    let vecs: Vec<Vector> = vec![Vector::from(chunk)];
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = $2, embeddings_updated_at = NOW() \
         WHERE id = $3",
    )
    .bind(&vecs)
    .bind(model)
    .bind(id)
    .execute(pool)
    .await
    .expect("stamp kb embedding");
}

async fn kb_search(pool: &PgPool, user: &str, query: &str, model: &str) -> Vec<String> {
    let store = PgKnowledgeBaseStore::new(pool.clone(), KnowledgeDeletePolicy::default());
    with_user_id(UserId::new(user), async {
        store
            .search(query, query_vec(), model, None, None, 10)
            .await
    })
    .await
    .expect("kb search must not raise on a mixed-model table")
    .entries
    .into_iter()
    .map(|e| e.id)
    .collect()
}

#[tokio::test]
async fn knowledge_search_returns_current_model_rows_when_table_holds_two_dimensions() {
    let Some(fx) = fixture("m684kb").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-current", "widget planning notes").await;
    stamp_kb(&fx.pool, "kb-current", query_vec(), Some(&current())).await;
    write_kb(&fx.pool, USER, "kb-old", "unrelated prose xyzzy").await;
    stamp_kb(
        &fx.pool,
        "kb-old",
        other_dimension_vec(),
        Some(&superseded()),
    )
    .await;

    let ids = kb_search(&fx.pool, USER, "nomatchterm", &current()).await;

    assert!(
        ids.contains(&"kb-current".to_string()),
        "the current-model row must still be reachable through the vector arm; got {ids:?}"
    );
    assert!(
        !ids.contains(&"kb-old".to_string()),
        "a 4-dimension row cannot be compared against a 3-dimension query; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_mid_reindex_returns_the_migrated_rows() {
    // A reindex updates in place, batch by batch, so the table legitimately
    // holds both models at once until the last batch lands.
    let Some(fx) = fixture("m684kbmid").await else {
        return;
    };
    for (id, migrated) in [("kb-a", true), ("kb-b", false), ("kb-c", true)] {
        write_kb(&fx.pool, USER, id, "shared corpus text").await;
        if migrated {
            stamp_kb(&fx.pool, id, query_vec(), Some(&current())).await;
        } else {
            stamp_kb(&fx.pool, id, other_dimension_vec(), Some(&superseded())).await;
        }
    }

    let ids = kb_search(&fx.pool, USER, "nomatchterm", &current()).await;

    assert!(
        ids.contains(&"kb-a".to_string()) && ids.contains(&"kb-c".to_string()),
        "already-migrated rows must be searchable mid-reindex; got {ids:?}"
    );
    assert!(
        !ids.contains(&"kb-b".to_string()),
        "the not-yet-migrated row must be invisible, not fatal; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_excludes_rows_stamped_with_a_superseded_model() {
    // Same dimension, different model: nothing raises, so only the predicate
    // stops the row being ranked as if it were comparable.
    let Some(fx) = fixture("m684kbsup").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-old", "unrelated prose xyzzy").await;
    stamp_kb(&fx.pool, "kb-old", query_vec(), Some(&superseded())).await;

    let ids = kb_search(&fx.pool, USER, "nomatchterm", &current()).await;

    assert!(
        ids.is_empty(),
        "a superseded-model row must be excluded, not silently ranked; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_keeps_a_renamed_model_with_an_unchanged_digest_visible() {
    // #655: the sweep restamps these rows rather than discarding them, so the
    // search must see them in the window before it runs -- otherwise a cosmetic
    // rename blanks vector search until the sweep catches up.
    let Some(fx) = fixture("m684kbren").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-renamed", "unrelated prose xyzzy").await;
    stamp_kb(&fx.pool, "kb-renamed", query_vec(), Some(&renamed())).await;

    let ids = kb_search(&fx.pool, USER, "nomatchterm", &current()).await;

    assert_eq!(
        ids,
        vec!["kb-renamed".to_string()],
        "same digest means the same model; the vector is usable as-is"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_excludes_a_stamp_carrying_no_digest() {
    // No digest on the stored side is no proof of sameness, matching how
    // `invalidate_stale_embeddings` treats the same row.
    let Some(fx) = fixture("m684kbnodig").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-bare", "unrelated prose xyzzy").await;
    stamp_kb(&fx.pool, "kb-bare", query_vec(), Some("nomic-embed-text")).await;

    let ids = kb_search(&fx.pool, USER, "nomatchterm", &current()).await;

    assert!(
        ids.is_empty(),
        "a bare-name stamp proves nothing about the model that wrote it; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_excludes_a_vector_with_no_model_stamp() {
    let Some(fx) = fixture("m684kbnull").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-unstamped", "unrelated prose xyzzy").await;
    stamp_kb(&fx.pool, "kb-unstamped", other_dimension_vec(), None).await;

    let ids = kb_search(&fx.pool, USER, "nomatchterm", &current()).await;

    assert!(
        ids.is_empty(),
        "a vector of unknown provenance has an unknown dimension; got {ids:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_full_text_arm_still_finds_rows_of_a_superseded_model() {
    // The model predicate belongs to the vector arm alone. Applied to the
    // full-text arm it would make content unfindable after a model change
    // instead of degrading to lexical search.
    let Some(fx) = fixture("m684kbfts").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-old", "quantum widget engine").await;
    stamp_kb(
        &fx.pool,
        "kb-old",
        other_dimension_vec(),
        Some(&superseded()),
    )
    .await;

    let ids = kb_search(&fx.pool, USER, "quantum widget", &current()).await;

    assert_eq!(
        ids,
        vec!["kb-old".to_string()],
        "a superseded-model row must stay findable lexically"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_with_an_empty_embedding_falls_back_to_full_text() {
    // The documented no-embedding fallback (a wedged embedding backend) is
    // unchanged: it is full-text-only and model-blind.
    let Some(fx) = fixture("m684kbempty").await else {
        return;
    };
    write_kb(&fx.pool, USER, "kb-old", "quantum widget engine").await;
    stamp_kb(
        &fx.pool,
        "kb-old",
        other_dimension_vec(),
        Some(&superseded()),
    )
    .await;

    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
    let hits = with_user_id(UserId::new(USER), async {
        store
            .search("quantum widget", Vec::new(), &current(), None, None, 10)
            .await
    })
    .await
    .expect("empty-embedding search");

    assert_eq!(
        hits.entries
            .iter()
            .map(|e| e.id.as_str())
            .collect::<Vec<_>>(),
        vec!["kb-old"],
        "no query vector means no vector arm, so no model scoping applies"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_model_predicate_does_not_leak_across_users() {
    // The model predicate is additional to user scoping, never a substitute:
    // another tenant's current-model row must stay invisible.
    let Some(fx) = fixture("m684kbtenant").await else {
        return;
    };
    write_kb(&fx.pool, "alice", "kb-alice", "alpha widget notes").await;
    stamp_kb(&fx.pool, "kb-alice", query_vec(), Some(&current())).await;
    write_kb(&fx.pool, "bob", "kb-bob", "beta gadget notes").await;
    stamp_kb(
        &fx.pool,
        "kb-bob",
        other_dimension_vec(),
        Some(&superseded()),
    )
    .await;

    let ids = kb_search(&fx.pool, "bob", "nomatchterm", &current()).await;

    assert!(
        !ids.contains(&"kb-alice".to_string()),
        "bob must not see alice's row just because its model matches; got {ids:?}"
    );
    fx.cleanup().await;
}

// --- skill index ------------------------------------------------------------

fn skill(name: &str, description: &str, body: &str) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: description.to_string(),
        kind: SkillKind::Skill,
        disk_path: format!("/usr/share/adelie/skills/{name}/SKILL.md"),
        owner_user_id: None,
        locality: Locality::Daemon,
        content_hash: format!("hash-{name}"),
        trust_tier: TrustTier::Local,
        source: Some("system".to_string()),
        tags: vec!["ops".to_string()],
        attachments: vec![],
        body: body.to_string(),
        metadata: serde_json::json!({}),
        present_on_disk: true,
        last_seen_at: None,
        approved_at: None,
        approved_by: None,
    }
}

async fn seed_skill(pool: &PgPool, s: IndexedSkill, chunk: Vec<f32>, model: &str) {
    let store = PgSkillIndexStore::new(pool.clone());
    let name = s.name.clone();
    reconcile_scan(&store, &SkillScope::Global, vec![s], chrono::Utc::now())
        .await
        .expect("seed skill");
    let vecs: Vec<Vector> = vec![Vector::from(chunk)];
    sqlx::query(
        "UPDATE skill_index SET embedding = $1::vector[], embedding_model = $2 \
         WHERE name = $3 AND owner_key = ''",
    )
    .bind(&vecs)
    .bind(model)
    .bind(&name)
    .execute(pool)
    .await
    .expect("stamp skill embedding");
}

async fn skill_search(pool: &PgPool, query: &str, model: &str) -> Vec<String> {
    PgSkillIndexStore::new(pool.clone())
        .search(query, query_vec(), model, 10)
        .await
        .expect("skill search must not raise on a mixed-model table")
        .into_iter()
        .map(|s| s.name)
        .collect()
}

#[tokio::test]
async fn skill_search_returns_current_model_rows_when_table_holds_two_dimensions() {
    let Some(fx) = fixture("m684si").await else {
        return;
    };
    seed_skill(
        &fx.pool,
        skill("invoice-run", "generate monthly invoices", "billing prose"),
        query_vec(),
        &current(),
    )
    .await;
    seed_skill(
        &fx.pool,
        skill("deploy-blog", "publish the blog", "static site"),
        other_dimension_vec(),
        &superseded(),
    )
    .await;

    let names = skill_search(&fx.pool, "nomatchterm", &current()).await;

    assert_eq!(
        names,
        vec!["invoice-run".to_string()],
        "the current-model skill must be reachable through the vector arm while a \
         differently-dimensioned row sits in the same table"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn skill_search_excludes_rows_stamped_with_a_superseded_model() {
    let Some(fx) = fixture("m684sisup").await else {
        return;
    };
    seed_skill(
        &fx.pool,
        skill("deploy-blog", "publish the blog", "static site"),
        query_vec(),
        &superseded(),
    )
    .await;

    let names = skill_search(&fx.pool, "nomatchterm", &current()).await;

    assert!(
        names.is_empty(),
        "a superseded-model skill must be excluded, not silently ranked; got {names:?}"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn skill_search_keeps_a_renamed_model_with_an_unchanged_digest_visible() {
    let Some(fx) = fixture("m684siren").await else {
        return;
    };
    seed_skill(
        &fx.pool,
        skill("invoice-run", "generate monthly invoices", "billing prose"),
        query_vec(),
        &renamed(),
    )
    .await;

    let names = skill_search(&fx.pool, "nomatchterm", &current()).await;

    assert_eq!(
        names,
        vec!["invoice-run".to_string()],
        "same digest means the same model; the vector is usable as-is"
    );
    fx.cleanup().await;
}

#[tokio::test]
async fn skill_full_text_arm_still_finds_rows_of_a_superseded_model() {
    let Some(fx) = fixture("m684sifts").await else {
        return;
    };
    seed_skill(
        &fx.pool,
        skill("invoice-run", "generate monthly invoices", "billing prose"),
        other_dimension_vec(),
        &superseded(),
    )
    .await;

    let names = skill_search(&fx.pool, "invoices", &current()).await;

    assert_eq!(
        names,
        vec!["invoice-run".to_string()],
        "a superseded-model skill must stay findable lexically"
    );
    fx.cleanup().await;
}

// --- tag registry -----------------------------------------------------------

/// Embeds every text to the same 3-dimension vector, so the dedup search is
/// decided by what the registry holds rather than by the embedding.
fn embed_to(vector: Vec<f32>) -> BackfillEmbedFn {
    Box::new(move |texts: Vec<String>| {
        let vector = vector.clone();
        Box::pin(async move { Ok(texts.iter().map(|_| vector.clone()).collect()) })
    })
}

/// Insert a tag straight into the registry with a chosen vector and stamp,
/// bypassing `create_or_match_tag` so the row's model is under test control.
async fn seed_tag(pool: &PgPool, name: &str, chunk: Vec<f32>, model: &str) {
    let vector = Vector::from(chunk);
    with_user_id(UserId::new(USER), async {
        sqlx::query(
            "INSERT INTO tag_registry \
                (user_id, name, description, examples, distinguish_from, embedding, embedding_model) \
             VALUES ($1, $2, 'seeded', '[]'::jsonb, '{}', $3, $4)",
        )
        .bind(USER)
        .bind(name)
        .bind(&vector)
        .bind(model)
        .execute(pool)
        .await
        .expect("seed tag");
    })
    .await;
}

async fn propose_tag(pool: &PgPool, name: &str, model: &str) -> CreateTagOutcome {
    with_user_id(UserId::new(USER), async {
        create_or_match_tag(
            pool,
            &embed_to(query_vec()),
            model,
            TagProposal {
                name: name.to_string(),
                description: "a proposed concept".to_string(),
                examples: vec![],
                distinguish_from: vec![],
            },
        )
        .await
    })
    .await
    .expect("tag dedup search must not raise on a mixed-model registry")
}

#[tokio::test]
async fn tag_dedup_search_survives_a_registry_holding_two_dimensions() {
    let Some(fx) = fixture("m684tag").await else {
        return;
    };
    seed_tag(
        &fx.pool,
        "old-concept",
        other_dimension_vec(),
        &superseded(),
    )
    .await;

    let outcome = propose_tag(&fx.pool, "new-concept", &current()).await;

    match outcome {
        CreateTagOutcome::Created(rec) => assert_eq!(rec.name, "new-concept"),
        CreateTagOutcome::RedirectedTo { existing, .. } => {
            panic!("must not redirect to a row it cannot compare against: {existing:?}")
        }
    }
    fx.cleanup().await;
}

#[tokio::test]
async fn tag_dedup_search_ignores_a_near_duplicate_from_a_superseded_model() {
    // Same dimension, so the distance is computable and would fall inside the
    // dedup threshold -- only the model predicate keeps it out.
    let Some(fx) = fixture("m684tagsup").await else {
        return;
    };
    seed_tag(&fx.pool, "old-concept", query_vec(), &superseded()).await;

    let outcome = propose_tag(&fx.pool, "new-concept", &current()).await;

    match outcome {
        CreateTagOutcome::Created(rec) => assert_eq!(rec.name, "new-concept"),
        CreateTagOutcome::RedirectedTo {
            existing, distance, ..
        } => panic!(
            "a superseded-model vector must not decide dedup: redirected to {} at {distance}",
            existing.name
        ),
    }
    fx.cleanup().await;
}

#[tokio::test]
async fn tag_dedup_search_keeps_a_renamed_model_with_an_unchanged_digest_visible() {
    let Some(fx) = fixture("m684tagren").await else {
        return;
    };
    seed_tag(&fx.pool, "old-concept", query_vec(), &renamed()).await;

    let outcome = propose_tag(&fx.pool, "new-concept", &current()).await;

    match outcome {
        CreateTagOutcome::RedirectedTo { existing, .. } => {
            assert_eq!(
                existing.name, "old-concept",
                "same digest means the same model; the stored vector still decides dedup"
            );
        }
        CreateTagOutcome::Created(rec) => panic!(
            "a renamed-but-identical model must not blind the dedup check; created {}",
            rec.name
        ),
    }
    fx.cleanup().await;
}
