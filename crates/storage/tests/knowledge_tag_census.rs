//! Integration coverage for the knowledge-base tag census (issue #1068).
//!
//! `PgKnowledgeBaseStore::search` reports two extra fields: how large the scope
//! it searched is (`scope_size`) and which tags that scope carries
//! (`available_tags`). Both come from one capped aggregate, so the properties
//! that matter are the cap, the recency of the sample, the ordering, and - most
//! of all - the `user_id` scoping: an unscoped census hands one tenant's tag
//! vocabulary, project and person names included, to every other tenant.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_tag_census
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use std::sync::Arc;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::{
    AVAILABLE_TAGS_LIMIT, KNOWLEDGE_TAG_CENSUS_SAMPLE, KnowledgeBaseStore, ScopeSize,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, run_migrations, with_user_id};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

/// RAII fixture: private schema, pool pinned to it, migrations applied.
struct Fixture {
    pool: PgPool,
    schema: String,
    admin_url: String,
}

impl Fixture {
    async fn try_new() -> Option<Self> {
        let url = support::test_database_url()?;
        let schema = format!("kbcensus_{}", Uuid::now_v7().simple());

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
            .max_connections(8)
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

/// The model a search passes. No row in this suite carries an embedding, so the
/// vector arm never contributes; the census is what every test reads.
const MODEL: &str = "test-model";

/// Bulk-insert `count` rows for `user`, all carrying `tags`, with `created_at`
/// running forwards from `first_epoch_secs`.
///
/// Writes go in raw rather than through `KnowledgeBaseStore::write` because the
/// sample cap needs more than a thousand rows, and because `created_at` has to
/// be controlled: the census samples the most recent entries, so a test that
/// let every row stamp `NOW()` could not tell recency from insertion order.
async fn seed(
    pool: &PgPool,
    user: &str,
    id_prefix: &str,
    count: i64,
    tags: &[&str],
    first_epoch_secs: i64,
) {
    let tags: Vec<String> = tags.iter().map(|t| (*t).to_string()).collect();
    sqlx::query(
        "INSERT INTO knowledge_base (id, user_id, content, tags, metadata, created_at, updated_at)
         SELECT $1 || '-' || i, $2, 'seeded entry ' || i, $3::text[], '{}'::jsonb,
                to_timestamp($4 + i), to_timestamp($4 + i)
         FROM generate_series(1, $5) AS i",
    )
    .bind(id_prefix)
    .bind(user)
    .bind(&tags)
    .bind(first_epoch_secs)
    .bind(count)
    .execute(pool)
    .await
    .expect("seed knowledge rows");
}

/// Write one entry through the store, so the row takes the same normalization
/// path a real write takes.
async fn write_entry(
    store: &PgKnowledgeBaseStore,
    user: &str,
    id: &str,
    content: &str,
    tags: &[&str],
) {
    let entry = KnowledgeEntry::new(id, content, tags.iter().map(|t| (*t).to_string()).collect());
    with_user_id(UserId::new(user), async {
        store.write(entry).await.expect("write entry");
    })
    .await;
}

// -- the sample: capped, and taken from the most recent entries ---------------

#[tokio::test]
async fn tag_census_samples_the_thousand_most_recent_entries() {
    // The cap keeps one extra aggregate per search from becoming a full scan,
    // and `ORDER BY created_at DESC` decides WHICH rows the cap keeps.
    //
    // The five oldest rows are inserted FIRST, so they sit at the head of the
    // heap. MUTATION: drop the `ORDER BY created_at DESC` from the census and a
    // bare `LIMIT` reads them in heap order, which puts `topic:ancient` inside
    // the sample -> this test goes RED. Dropping the `LIMIT` altogether does
    // the same.
    with_fixture(
        "tag_census_samples_the_thousand_most_recent_entries",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed(&fx.pool, "alice", "old", 5, &["topic:ancient"], 1_000).await;
            seed(
                &fx.pool,
                "alice",
                "new",
                KNOWLEDGE_TAG_CENSUS_SAMPLE as i64,
                &["topic:recent"],
                2_000_000,
            )
            .await;

            let page = with_user_id(UserId::new("alice"), async {
                store
                    .search("seeded", Vec::new(), MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("search");

            assert!(
                page.available_tags.contains(&"topic:recent".to_string()),
                "premise: the recent rows are in scope; got {:?}",
                page.available_tags
            );
            assert!(
                !page.available_tags.contains(&"topic:ancient".to_string()),
                "the census must stop after the {KNOWLEDGE_TAG_CENSUS_SAMPLE} most recent \
                 entries, so the older rows beyond the cap cannot contribute tags; got {:?}",
                page.available_tags
            );
            assert_eq!(
                page.scope_size,
                ScopeSize::Many,
                "a scope that reached the sample cap is at least that large, so it is MANY"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn tag_census_sample_ordering_is_total() {
    // `created_at` alone does not order the sample: rows that share one
    // timestamp are cut apart by whatever secondary order the plan happens to
    // produce, and that order follows the rows' physical position. So an edit
    // anywhere in the table can change which tags a search reports, and the
    // model sees the vocabulary churn with no cause it can act on. `id` breaks
    // the tie, and ids are unique, so the order is total.
    //
    // Every row here shares one `created_at`, and ten rows carry `topic:cut`.
    // Under `created_at DESC, id DESC` those ten hold the lowest ids, so the
    // sample cap always drops them.
    //
    // MUTATION: drop `, id DESC` from the census. The rows carrying
    // `topic:kept` are then rewritten between the two searches, which moves
    // them behind the untouched `topic:cut` rows in physical order, so
    // `topic:cut` enters the sample and the second run reports a tag the first
    // did not -> this test goes RED.
    with_fixture("tag_census_sample_ordering_is_total", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
        let cut = 10_i64;
        let total = KNOWLEDGE_TAG_CENSUS_SAMPLE as i64 + cut;

        sqlx::query(
            "INSERT INTO knowledge_base
                 (id, user_id, content, tags, metadata, created_at, updated_at)
             SELECT 'tie-' || lpad(i::text, 4, '0'), 'alice', 'tied entry ' || i,
                    CASE WHEN i <= $1 THEN ARRAY['topic:cut'] ELSE ARRAY['topic:kept'] END,
                    '{}'::jsonb, to_timestamp(1000), to_timestamp(1000)
             FROM generate_series(1, $2) AS i",
        )
        .bind(cut)
        .bind(total)
        .execute(&fx.pool)
        .await
        .expect("seed tied rows");

        let census = |store: PgKnowledgeBaseStore| async move {
            with_user_id(UserId::new("alice"), async move {
                store
                    .search("tied", Vec::new(), MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("search")
        };

        let first = census(PgKnowledgeBaseStore::new(
            fx.pool.clone(),
            KnowledgeDeletePolicy::default(),
        ))
        .await;
        assert_eq!(
            first.available_tags,
            vec!["topic:kept".to_string()],
            "the id tiebreak drops the lowest ids, which are the only `topic:cut` rows"
        );
        assert_eq!(
            first.scope_size,
            ScopeSize::Many,
            "a scope that reached the sample cap is at least that large"
        );

        // Rewrite the kept rows. Each update writes a new tuple version at the
        // end of the heap, so the untouched `topic:cut` rows now sit first in
        // physical order - the exact reordering a `VACUUM` or an unrelated edit
        // produces in a live store.
        sqlx::query(
            "UPDATE knowledge_base SET content = content
             WHERE user_id = 'alice' AND tags && ARRAY['topic:kept']",
        )
        .execute(&fx.pool)
        .await
        .expect("rewrite the kept rows");

        let second = census(store).await;
        assert_eq!(
            second.available_tags, first.available_tags,
            "the same scope must report the same tags after the rows move"
        );
        assert_eq!(second.scope_size, first.scope_size);
        fx
    })
    .await;
}

// -- the scoping: one tenant's vocabulary never reaches another --------------

#[tokio::test]
async fn tag_census_never_returns_another_users_tags() {
    // Tag names carry project and person names, so an unscoped census is a
    // disclosure, not just a wrong number.
    //
    // MUTATION: drop `WHERE user_id = $1` from the census and Bob's response
    // carries `person:alice-doctor` and `project:alice-secret`, and reports
    // MANY instead of FEW -> this test goes RED on both assertions.
    with_fixture(
        "tag_census_never_returns_another_users_tags",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed(
                &fx.pool,
                "alice",
                "alice",
                40,
                &["project:alice-secret", "person:alice-doctor"],
                1_000,
            )
            .await;
            write_entry(&store, "bob", "kb-bob", "bob's own note", &["topic:bob"]).await;

            let page = with_user_id(UserId::new("bob"), async {
                store
                    .search("note", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("bob search");

            assert_eq!(
                page.available_tags,
                vec!["topic:bob".to_string()],
                "bob's census must report only bob's own tags"
            );
            assert_eq!(
                page.scope_size,
                ScopeSize::Few,
                "bob owns one entry, so his scope is FEW - alice's forty are not his scope"
            );
            fx
        },
    )
    .await;
}

// -- ordering, cap, and filters ---------------------------------------------

#[tokio::test]
async fn tag_census_orders_available_tags_by_frequency_then_name() {
    // The order carries the whole signal, because no counts travel with the
    // tags. `rare` is alphabetically between the two common tags, so a census
    // that sorted by name alone would put it in the middle.
    with_fixture(
        "tag_census_orders_available_tags_by_frequency_then_name",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed(
                &fx.pool,
                "alice",
                "common",
                5,
                &["aaa-common", "zzz-common"],
                1_000,
            )
            .await;
            seed(&fx.pool, "alice", "rare", 1, &["rare"], 2_000).await;

            let page = with_user_id(UserId::new("alice"), async {
                store
                    .search("seeded", Vec::new(), MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("search");

            assert_eq!(
                page.available_tags,
                vec![
                    "aaa-common".to_string(),
                    "zzz-common".to_string(),
                    "rare".to_string(),
                ],
                "frequency decides first, tag name breaks the tie"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn tag_census_caps_available_tags_at_fifty() {
    // The list travels to a language model inside a tool result, so an
    // unbounded vocabulary spends context without adding signal.
    with_fixture("tag_census_caps_available_tags_at_fifty", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
        let over_cap = AVAILABLE_TAGS_LIMIT + 10;
        for i in 0..over_cap {
            let tag = format!("topic:t{i:03}");
            seed(
                &fx.pool,
                "alice",
                &format!("row{i:03}"),
                1,
                &[tag.as_str()],
                1_000 + i as i64,
            )
            .await;
        }

        let page = with_user_id(UserId::new("alice"), async {
            store
                .search("seeded", Vec::new(), MODEL, None, None, 10)
                .await
        })
        .await
        .expect("search");

        assert_eq!(page.available_tags.len(), AVAILABLE_TAGS_LIMIT);
        let expected: Vec<String> = (0..AVAILABLE_TAGS_LIMIT)
            .map(|i| format!("topic:t{i:03}"))
            .collect();
        assert_eq!(
            page.available_tags, expected,
            "every tag is carried once, so the name tiebreak decides which fifty survive"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn tag_census_honours_include_and_exclude_filters() {
    // The filters define the scope. The include filter arrives differently
    // cased on purpose: the census must normalize it the way the search itself
    // does, or it reports a vocabulary for a scope nobody searched.
    with_fixture(
        "tag_census_honours_include_and_exclude_filters",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            write_entry(&store, "alice", "kb-a", "alpha note", &["keep", "alpha"]).await;
            write_entry(&store, "alice", "kb-b", "beta note", &["keep", "beta"]).await;
            write_entry(&store, "alice", "kb-c", "gamma note", &["drop", "gamma"]).await;

            let page = with_user_id(UserId::new("alice"), async {
                store
                    .search(
                        "note",
                        Vec::new(),
                        MODEL,
                        Some(vec!["KEEP".into()]),
                        Some(vec!["Beta".into()]),
                        10,
                    )
                    .await
            })
            .await
            .expect("filtered search");

            assert_eq!(
                page.available_tags,
                vec!["alpha".to_string(), "keep".to_string()],
                "only the one entry passing both filters may contribute tags"
            );
            assert_eq!(page.scope_size, ScopeSize::Few);
            fx
        },
    )
    .await;
}

// -- the degraded path reports the same fields -------------------------------

#[tokio::test]
async fn knowledge_search_fallback_reports_scope_and_tags() {
    // An empty query embedding (the embedding backend timed out) sends the
    // search down the full-text-only path. That costs semantic recall; it must
    // not cost the caller the scope report as well.
    //
    // The scope holds three entries and the query matches one, which also pins
    // the settled contract: `scope_size` describes the SCOPE, never the number
    // of entries that matched the query.
    with_fixture(
        "knowledge_search_fallback_reports_scope_and_tags",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            write_entry(
                &store,
                "alice",
                "kb-1",
                "quokka sighting",
                &["topic:wildlife"],
            )
            .await;
            write_entry(
                &store,
                "alice",
                "kb-2",
                "marmot sighting",
                &["topic:wildlife"],
            )
            .await;
            write_entry(
                &store,
                "alice",
                "kb-3",
                "deploy runbook",
                &["project:adelie-ai"],
            )
            .await;

            let page = with_user_id(UserId::new("alice"), async {
                store
                    .search("quokka", Vec::new(), MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("fallback search");

            assert_eq!(
                page.entries
                    .iter()
                    .map(|e| e.id.as_str())
                    .collect::<Vec<_>>(),
                vec!["kb-1"],
                "premise: the full-text arm matched one entry"
            );
            assert_eq!(
                page.available_tags,
                vec![
                    "topic:wildlife".to_string(),
                    "project:adelie-ai".to_string()
                ],
                "the fallback path must report the whole scope's tags, not the matched entry's"
            );
            assert_eq!(page.scope_size, ScopeSize::Few);
            fx
        },
    )
    .await;
}
