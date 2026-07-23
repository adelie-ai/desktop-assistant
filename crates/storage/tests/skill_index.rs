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
use desktop_assistant_core::ports::skill_index::{SkillIndexStore, conformance};
use desktop_assistant_core::skill_catalog::reconcile_scan;
use desktop_assistant_storage::embedding_backfill::{BackfillEmbedFn, backfill_skill_embeddings};
use desktop_assistant_storage::{PgSkillIndexStore, run_migrations};
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
    reconcile_scan(
        store,
        &SkillScope::Global,
        skills,
        conformance::first_scan_at(),
    )
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
        let hits = store.search("invoice", vec![], 10).await.expect("search");
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
