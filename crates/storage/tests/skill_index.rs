//! Integration coverage for `PgSkillIndexStore` and `backfill_skill_embeddings`
//! (#573, #639).
//!
//! Catalog semantics come from the shared `SkillIndexStore` contract in
//! `core::ports::skill_index::conformance`, run here against a real Postgres --
//! one test per case, so a failure names the broken guarantee and not just this
//! adapter. The tests below the contract block cover what is genuinely local to
//! Postgres: embedding preservation across an unchanged-hash rescan (SQLite has
//! no vector column), hybrid/full-text search, and the embedding backfill.
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips (loudly, via
//! `support`).

mod support;

use std::sync::Arc;

use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, SkillScope, TrustTier};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::skill_index::{SkillIndexStore, conformance};
use desktop_assistant_core::skill_catalog::reconcile_scan;
use desktop_assistant_storage::embedding_backfill::{BackfillEmbedFn, backfill_skill_embeddings};
use desktop_assistant_storage::{PgSkillIndexStore, run_migrations};
use pgvector::Vector;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("issue573si_{}", Uuid::now_v7().simple());

        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("connect to TEST_DATABASE_URL");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA \"{schema}\"")))
            .execute(&admin)
            .await
            .expect("create test schema");
        admin.close().await;

        let schema_for_hook = Arc::new(schema.clone());
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .after_connect(move |conn, _| {
                let schema = Arc::clone(&schema_for_hook);
                Box::pin(async move {
                    let sql = format!("SET search_path TO \"{schema}\", public");
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&url)
            .await
            .expect("connect per-test pool");

        run_migrations(&pool)
            .await
            .expect("run_migrations succeeds");

        Some(Self {
            pool,
            schema,
            admin_url: url,
        })
    }

    async fn cleanup(self) {
        self.pool.close().await;
        if let Ok(admin) = PgPoolOptions::new()
            .max_connections(1)
            .connect(&self.admin_url)
            .await
        {
            let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                "DROP SCHEMA \"{}\" CASCADE",
                self.schema
            )))
            .execute(&admin)
            .await;
            admin.close().await;
        }
    }
}

async fn with_fixture<F, Fut>(name: &str, body: F)
where
    F: FnOnce(Fixture) -> Fut,
    Fut: std::future::Future<Output = Fixture>,
{
    let Some(fx) = Fixture::try_new().await else {
        eprintln!("skip: TEST_DATABASE_URL not set; {name} pass-skipped");
        return;
    };
    let fx = body(fx).await;
    fx.cleanup().await;
}

fn skill(name: &str, description: &str, hash: &str, body: &str) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: description.to_string(),
        kind: if body.contains("## Steps") {
            SkillKind::Workflow
        } else {
            SkillKind::Skill
        },
        disk_path: format!("/usr/share/adelie/skills/{name}/SKILL.md"),
        owner_user_id: None,
        locality: Locality::Daemon,
        content_hash: hash.to_string(),
        trust_tier: TrustTier::Local,
        source: Some("system".to_string()),
        tags: vec!["ops".to_string()],
        attachments: vec![],
        body: body.to_string(),
        metadata: serde_json::json!({"author": "test"}),
        present_on_disk: true,
        last_seen_at: None,
    }
}

/// A user-scoped skill, mirroring [`skill`] but owned by `owner` (a
/// synthetic tenant id such as `"tenant-a"`, never a real identity).
fn owned_skill(name: &str, owner: &str, description: &str, hash: &str, body: &str) -> IndexedSkill {
    IndexedSkill {
        name: name.to_string(),
        description: description.to_string(),
        kind: SkillKind::Skill,
        disk_path: format!("/home/{owner}/.local/share/adele/skills/{name}/SKILL.md"),
        owner_user_id: Some(owner.to_string()),
        locality: Locality::Client,
        content_hash: hash.to_string(),
        trust_tier: TrustTier::Local,
        source: Some("client".to_string()),
        tags: vec!["ops".to_string()],
        attachments: vec![],
        body: body.to_string(),
        metadata: serde_json::json!({"author": "test"}),
        present_on_disk: true,
        last_seen_at: None,
    }
}

fn fake_embed_fn() -> BackfillEmbedFn {
    // Deterministic fixed-dimension vector per input text.
    Box::new(|texts: Vec<String>| {
        Box::pin(async move { Ok(texts.iter().map(|_| vec![0.1_f32, 0.2, 0.3, 0.4]).collect()) })
    })
}

/// Seed a global scan through the reconcile pass, at the contract's fixed
/// instant, so adapter-specific tests exercise the same write path production
/// uses.
async fn seed(store: &PgSkillIndexStore, skills: Vec<IndexedSkill>) {
    seed_scope(store, &SkillScope::Global, skills).await;
}

/// Like [`seed`], but scoped to any [`SkillScope`] -- for tests that seed a
/// specific owner rather than the global scope.
async fn seed_scope(store: &PgSkillIndexStore, scope: &SkillScope, skills: Vec<IndexedSkill>) {
    reconcile_scan(store, scope, skills, conformance::first_scan_at())
        .await
        .expect("seed scan");
}

/// One test per contract case, each against its own throwaway schema.
macro_rules! conformance_tests {
    ($($case:ident),+ $(,)?) => {
        $(
            #[tokio::test]
            async fn $case() {
                with_fixture(stringify!($case), |fx| async move {
                    conformance::$case(&PgSkillIndexStore::new(fx.pool.clone())).await;
                    fx
                })
                .await;
            }
        )+
    };
}

conformance_tests!(
    removed_skill_survives_reconcile,
    empty_scan_preserves_the_catalog,
    unseen_skill_keeps_its_last_seen_at,
    rescan_restores_presence_when_skill_returns,
    reconcile_leaves_other_scopes_untouched,
    absent_skills_are_still_searchable,
    reconcile_is_idempotent,
    upsert_ignores_caller_supplied_presence,
    get_is_scope_addressed,
    set_presence_tolerates_unknown_and_empty,
);

#[tokio::test]
async fn reindex_preserves_embedding_when_hash_unchanged_and_nulls_it_on_change() {
    with_fixture("reindex_preserves_embedding", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed(&store, vec![skill("a", "desc", "hash-1", "body")]).await;

        // Simulate the backfill having embedded the row.
        sqlx::query(
            "UPDATE skill_index SET embedding = ARRAY['[1,2,3]']::vector[], \
             embedding_model = 'm1' WHERE name = 'a' AND owner_key = ''",
        )
        .execute(&fx.pool)
        .await
        .unwrap();

        // Rescan with the SAME hash: embedding is preserved.
        seed(&store, vec![skill("a", "desc updated", "hash-1", "body")]).await;
        let model: Option<String> =
            sqlx::query_scalar("SELECT embedding_model FROM skill_index WHERE name = 'a'")
                .fetch_one(&fx.pool)
                .await
                .unwrap();
        assert_eq!(
            model.as_deref(),
            Some("m1"),
            "unchanged hash keeps embedding"
        );

        // Rescan with a CHANGED hash: embedding is nulled for re-embedding.
        seed(&store, vec![skill("a", "desc", "hash-2", "body")]).await;
        let model: Option<String> =
            sqlx::query_scalar("SELECT embedding_model FROM skill_index WHERE name = 'a'")
                .fetch_one(&fx.pool)
                .await
                .unwrap();
        assert_eq!(model, None, "changed hash nulls embedding for re-embed");
        fx
    })
    .await;
}

#[tokio::test]
async fn fts_search_finds_by_keyword_and_get_is_owner_scoped() {
    with_fixture("fts_search", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed(
            &store,
            vec![
                skill(
                    "invoice-run",
                    "generate monthly invoices",
                    "h1",
                    "billing prose",
                ),
                skill("deploy-blog", "publish the blog", "h2", "static site"),
            ],
        )
        .await;

        // Empty embedding -> FTS-only path.
        let hits = store
            .search("invoice", vec![], "test-model", 10)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "invoice-run");

        // No user-scoped rows exist, so a user-scoped get misses.
        assert!(
            store
                .get("invoice-run", Some("nobody"))
                .await
                .unwrap()
                .is_none()
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn backfill_embeds_null_model_rows() {
    with_fixture("backfill", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed(
            &store,
            vec![
                skill("a", "alpha skill", "h1", "body a"),
                skill("b", "beta skill", "h2", "body b"),
            ],
        )
        .await;

        let updated = backfill_skill_embeddings(&fx.pool, &fake_embed_fn(), "test-model")
            .await
            .expect("backfill");
        assert_eq!(updated, 2, "both NULL-model rows embedded");

        let embedded: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM skill_index WHERE embedding IS NOT NULL AND embedding_model = 'test-model'",
        )
        .fetch_one(&fx.pool)
        .await
        .unwrap();
        assert_eq!(embedded, 2);

        // A second backfill with the same model is a no-op (nothing stale).
        let again = backfill_skill_embeddings(&fx.pool, &fake_embed_fn(), "test-model")
            .await
            .unwrap();
        assert_eq!(again, 0);
        fx
    })
    .await;
}

// -- #911: `get` must agree with `search`/`list` about whose rows a caller
// -- can read. Each case below is a named acceptance criterion from the
// -- issue. `get_refuses_another_users_skill`, `get_ignores_a_spoofed_owner_argument`,
// -- and `get_scoping_matches_search_and_list` are the ones that actually
// -- catch the vulnerability (they fail against the pre-fix `get`, which
// -- trusted the caller-supplied `owner` string outright); the other three
// -- assert behavior the fix must leave alone.

#[tokio::test]
async fn get_returns_a_global_skill() {
    with_fixture("get_returns_a_global_skill", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed(
            &store,
            vec![skill("changelog", "publish notes", "h1", "steps")],
        )
        .await;

        // A global skill (owner_user_id IS NULL) stays readable by any
        // caller, including one with a real, unrelated identity installed.
        let hit = with_user_id(UserId::new("someone"), async {
            store.get("changelog", None).await.unwrap()
        })
        .await;

        assert_eq!(
            hit.expect("a global skill is readable by any caller").body,
            "steps"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn get_returns_own_user_scoped_skill() {
    with_fixture("get_returns_own_user_scoped_skill", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed_scope(
            &store,
            &SkillScope::Owner("tenant-a".to_string()),
            vec![owned_skill(
                "deploy-notes",
                "tenant-a",
                "A's own procedure",
                "ha1",
                "A's steps",
            )],
        )
        .await;

        let hit = with_user_id(UserId::new("tenant-a"), async {
            store.get("deploy-notes", Some("tenant-a")).await.unwrap()
        })
        .await;

        assert_eq!(
            hit.expect("the owner reading their own skill must succeed")
                .body,
            "A's steps",
            "the normal path is unaffected by the fix"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn get_refuses_another_users_skill() {
    with_fixture("get_refuses_another_users_skill", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed_scope(
            &store,
            &SkillScope::Owner("tenant-a".to_string()),
            vec![owned_skill(
                "secret-recipe",
                "tenant-a",
                "A's private procedure",
                "ha2",
                "A's private steps",
            )],
        )
        .await;

        // Tenant B names tenant A's id as the `owner` argument -- exactly
        // what an LLM forwarding a caller-supplied tool argument could do.
        let hit = with_user_id(UserId::new("tenant-b"), async {
            store.get("secret-recipe", Some("tenant-a")).await.unwrap()
        })
        .await;

        assert!(
            hit.is_none(),
            "tenant B naming tenant A's id must get nothing back"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn get_ignores_a_spoofed_owner_argument() {
    with_fixture("get_ignores_a_spoofed_owner_argument", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed_scope(
            &store,
            &SkillScope::Owner("tenant-a".to_string()),
            vec![owned_skill(
                "deploy-notes",
                "tenant-a",
                "A's own procedure",
                "ha3",
                "A's steps",
            )],
        )
        .await;
        seed_scope(
            &store,
            &SkillScope::Owner("tenant-b".to_string()),
            vec![owned_skill(
                "deploy-notes",
                "tenant-b",
                "B's own procedure",
                "hb3",
                "B's steps",
            )],
        )
        .await;

        // Tenant B asks for "deploy-notes" but *names* tenant A as the
        // owner. The LLM-supplied string must not widen scope: B gets B's
        // own row for that name, never A's.
        let hit = with_user_id(UserId::new("tenant-b"), async {
            store.get("deploy-notes", Some("tenant-a")).await.unwrap()
        })
        .await;

        assert_eq!(
            hit.expect("B has an own-scoped row for this name").body,
            "B's steps",
            "a spoofed owner argument resolves to the caller's own row, never the named one"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn single_tenant_get_is_unaffected() {
    with_fixture("single_tenant_get_is_unaffected", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed(
            &store,
            vec![skill("changelog", "publish notes", "h4", "steps")],
        )
        .await;

        // No task-local identity installed at all -- the desktop,
        // single-tenant path. `current_user_id()` falls back to the schema
        // sentinel, and a global skill is still readable exactly as before.
        let hit = store.get("changelog", None).await.unwrap();
        assert_eq!(
            hit.expect("single-tenant global read is unaffected").body,
            "steps"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn get_scoping_matches_search_and_list() {
    with_fixture("get_scoping_matches_search_and_list", |fx| async move {
        let store = PgSkillIndexStore::new(fx.pool.clone());
        seed(
            &store,
            vec![skill(
                "global-runbook",
                "shared procedure",
                "hg5",
                "shared steps",
            )],
        )
        .await;
        seed_scope(
            &store,
            &SkillScope::Owner("tenant-a".to_string()),
            vec![owned_skill(
                "a-only",
                "tenant-a",
                "A's procedure",
                "ha5",
                "A's steps",
            )],
        )
        .await;
        seed_scope(
            &store,
            &SkillScope::Owner("tenant-b".to_string()),
            vec![owned_skill(
                "b-only",
                "tenant-b",
                "B's procedure",
                "hb5",
                "B's steps",
            )],
        )
        .await;

        with_user_id(UserId::new("tenant-a"), async {
            // search: global + A's own, never B's.
            let hits = store
                .search("procedure", vec![], "test-model", 10)
                .await
                .unwrap();
            let names: std::collections::BTreeSet<_> =
                hits.iter().map(|s| s.name.clone()).collect();
            assert!(
                names.contains("global-runbook"),
                "search sees the global row"
            );
            assert!(names.contains("a-only"), "search sees A's own row");
            assert!(
                !names.contains("b-only"),
                "search must not surface another tenant's skill"
            );

            // list: the same boundary.
            let listed = store.list(None).await.unwrap();
            let listed_names: std::collections::BTreeSet<_> =
                listed.iter().map(|s| s.name.clone()).collect();
            assert!(
                listed_names.contains("global-runbook"),
                "list sees the global row"
            );
            assert!(listed_names.contains("a-only"), "list sees A's own row");
            assert!(
                !listed_names.contains("b-only"),
                "list must not surface another tenant's skill"
            );

            // get: the same boundary, addressed one row at a time.
            assert!(
                store.get("global-runbook", None).await.unwrap().is_some(),
                "get sees the global row"
            );
            assert!(
                store
                    .get("a-only", Some("tenant-a"))
                    .await
                    .unwrap()
                    .is_some(),
                "get sees A's own row"
            );
            assert!(
                store
                    .get("b-only", Some("tenant-b"))
                    .await
                    .unwrap()
                    .is_none(),
                "get must not surface another tenant's skill either -- \
                 the three queries agree"
            );
        })
        .await;
        fx
    })
    .await;
}

// -- #1107 sweep: `search_hybrid`'s vector arm (`vr`) had the same
// -- `ROW_NUMBER() OVER (ORDER BY dist) ... LIMIT $4` shape as the knowledge
// -- base's vector arm, with no statement-level `ORDER BY` before the `LIMIT`.
// -- `tr` (the text arm) already carries `ORDER BY ts_rank_cd(...) DESC`
// -- before its own `LIMIT` and needed no change.

/// Stamp a `vector[]` embedding onto a global skill by name (mirrors
/// [`set_embedding`] in `knowledge_hybrid_and_pagination.rs`).
async fn set_skill_embedding(pool: &PgPool, name: &str, chunk: Vec<f32>) {
    let vecs: Vec<Vector> = vec![Vector::from(chunk)];
    sqlx::query(
        "UPDATE skill_index SET embedding = $1::vector[], embedding_model = 'test-model' \
         WHERE name = $2 AND owner_key = ''",
    )
    .bind(&vecs)
    .bind(name)
    .execute(pool)
    .await
    .expect("stamp skill embedding");
}

#[tokio::test]
async fn skill_search_vector_arm_truncates_to_the_nearest_candidates() {
    // Contract PIN, not a red-to-green reproduction -- same reasoning as
    // `vector_arm_truncates_to_the_nearest_candidates_not_an_arbitrary_subset`
    // in `knowledge_hybrid_and_pagination.rs`: `vr.rank_v` feeds `fused.score`,
    // which the outer `ORDER BY f.score DESC` reads, so the planner cannot
    // eliminate the window and today's plan preserves distance order into the
    // `LIMIT`. This test guards the property, not a live defect.
    //
    // The FTS query term ("zzznomatchzzz") matches no seeded skill, so `tr` is
    // empty and `fused` reduces to `vr` alone -- isolating the vector arm.
    with_fixture(
        "skill_search_vector_arm_truncates_to_the_nearest_candidates",
        |fx| async move {
            let store = PgSkillIndexStore::new(fx.pool.clone());

            // 20 skills -- more than fetch_limit (limit=6 -> fetch_limit=12).
            let skills: Vec<IndexedSkill> = (0..20u32)
                .map(|i| {
                    skill(
                        &format!("skill{i:02}"),
                        &format!("generic filler capability {i}"),
                        &format!("hash{i:02}"),
                        "filler body",
                    )
                })
                .collect();
            seed(&store, skills).await;

            // Embeddings at strictly increasing cosine distance from [1,0,0]:
            // skill00 is nearest, skill19 is farthest.
            for i in 0..20u32 {
                let f = i as f32 * 0.01;
                set_skill_embedding(&fx.pool, &format!("skill{i:02}"), vec![1.0 - f, f, 0.0]).await;
            }

            let hits = store
                .search("zzznomatchzzz", vec![1.0, 0.0, 0.0], "test-model", 6)
                .await
                .expect("search");
            let names: Vec<&str> = hits.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(
                names,
                vec![
                    "skill00", "skill01", "skill02", "skill03", "skill04", "skill05"
                ],
                "the vector arm's contribution must be exactly the 6 nearest \
                 skills, nearest first, out of 20 candidates and a \
                 fetch_limit of 12; got {names:?}"
            );
            fx
        },
    )
    .await;
}

// -- #1107 follow-up: `fused`'s `ORDER BY f.score DESC` has the same
// -- undefined-truncation defect the window-function fix addressed, one
// -- level out. RRF ties exactly by construction (a row found by ONLY one
// -- arm at rank 1 scores exactly `1/(60+1)`, whichever arm found it), so a
// -- score-only `ORDER BY` leaves which of two tied skills the `LIMIT` keeps
// -- undefined. Unlike the window-function sites, this tie is directly
// -- constructible, so this is a red-to-green reproduction, not a pin.

#[tokio::test]
async fn skill_search_truncates_to_a_defined_row_when_fused_scores_tie() {
    // "zzz-skill" is found ONLY by the text arm (it is the sole row
    // containing the FTS query term "gronk"). "aaa-skill" is found ONLY by
    // the vector arm (its embedding exactly matches the query vector). Both
    // therefore score exactly `1.0 / (60 + 1)` -- an exact IEEE-754 tie, not
    // an approximate one.
    //
    // `skill_index` has no single surrogate id column; its uniqueness is the
    // composite `(name, owner_key)` (`idx_skill_index_name_owner`), so that
    // composite -- not a recency column -- is the natural deterministic
    // tiebreak. Both rows are global (`owner_key` = `''`), so `name` alone
    // decides: `name ASC` must pick "aaa-skill" (the lexicographically
    // smaller name) when `limit` truncates the tied pair down to 1.
    //
    // This name assignment (not the reverse) is what makes the test a
    // genuine red-to-green: probed directly, the pre-fix query's
    // `FULL OUTER JOIN` between `vr` and `tr` emits the text-arm-only row
    // first when scores tie, regardless of which literal name carries which
    // role. Naming the text-matched row "zzz-skill" (the lexicographically
    // LARGER name) makes that incidental behavior disagree with `name ASC`.
    with_fixture(
        "skill_search_truncates_to_a_defined_row_when_fused_scores_tie",
        |fx| async move {
            let store = PgSkillIndexStore::new(fx.pool.clone());

            seed(
                &store,
                vec![
                    skill(
                        "zzz-skill",
                        "the distinctive gronk term appears here",
                        "h1",
                        "b1",
                    ),
                    skill(
                        "aaa-skill",
                        "unrelated prose that never matches",
                        "h2",
                        "b2",
                    ),
                ],
            )
            .await;
            set_skill_embedding(&fx.pool, "aaa-skill", vec![1.0, 0.0, 0.0]).await;
            // zzz-skill is left unembedded on purpose.

            let hits = store
                .search("gronk", vec![1.0, 0.0, 0.0], "test-model", 1)
                .await
                .expect("search");
            let names: Vec<&str> = hits.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(
                names,
                vec!["aaa-skill"],
                "with an exact fused-score tie, the truncation to 1 row must \
                 be decided by name ASC (\"aaa-skill\" < \"zzz-skill\"), not \
                 by an undefined physical row order; got {names:?}"
            );
            fx
        },
    )
    .await;
}
