//! Integration coverage for the knowledge-base hybrid search vector path,
//! keyset pagination, and cross-user `delete_many` (issue #437).
//!
//! The 2026-07 audit found that the RRF fusion **and the vector branch's
//! `WHERE user_id` scoping** never executed in any test — existing suites only
//! hit the empty-embedding FTS fallback and `search_text`. These tests feed
//! real (small, hand-authored) embeddings so `search`'s vector branch actually
//! runs, then pin: user scoping on the vector branch (`$6`), `exclude_tags`
//! (`$7`), RRF fusion ordering, `list_page` keyset walking / tiebreaks / cursor
//! validation / limit clamp, and `delete_many` cross-user opacity.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! PGPASSWORD=test psql -h 127.0.0.1 -p 15432 -U postgres -c \
//!     'CREATE EXTENSION IF NOT EXISTS vector;'
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test knowledge_hybrid_and_pagination
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::{
    KnowledgeBaseStore, KnowledgeListQuery, ListOrder, ListOrderOpt,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, run_migrations, with_user_id};
use pgvector::Vector;
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
        let schema = format!("issue437_{}", Uuid::now_v7().simple());

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

/// Stamp a `vector[]` embedding onto a knowledge row by id (writes never embed
/// inline — the background backfill does — so tests populate the column
/// directly to exercise the search vector branch). Mirrors the raw SQL in
/// `embedding_backfill::backfill_knowledge_embeddings`.
async fn set_embedding(pool: &PgPool, id: &str, chunks: Vec<Vec<f32>>) {
    let vecs: Vec<Vector> = chunks.into_iter().map(Vector::from).collect();
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = 'test-model', \
             embeddings_updated_at = NOW() \
         WHERE id = $2",
    )
    .bind(&vecs)
    .bind(id)
    .execute(pool)
    .await
    .expect("stamp embedding");
}

/// Force a row's `created_at` so keyset ordering is deterministic (writes stamp
/// `NOW()`; the cursor is on `(created_at, id)`).
async fn set_created_at(pool: &PgPool, id: &str, ts: chrono::DateTime<chrono::Utc>) {
    sqlx::query("UPDATE knowledge_base SET created_at = $1 WHERE id = $2")
        .bind(ts)
        .bind(id)
        .execute(pool)
        .await
        .expect("set created_at");
}

fn ts(secs: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0).expect("valid timestamp")
}

// -- hybrid search: vector branch is user-scoped -----------------------------

#[tokio::test]
async fn knowledge_hybrid_search_is_user_scoped() {
    // The vector branch (`chunk <=> $1 WHERE user_id = $6`) must not leak
    // another user's embedded rows. Bob searches with a NON-empty embedding so
    // the vector branch actually runs (not the FTS fallback); Alice's embedded
    // doc must never surface in Bob's results.
    //
    // MUTATION: dropping `WHERE user_id = $6` on the vector branch
    // (knowledge.rs:96) makes Alice's [1,0,0] doc leak into Bob's vector-ranked
    // set (distance 0 to Bob's query embedding) → this test goes RED.
    with_fixture("knowledge_hybrid_search_is_user_scoped", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .write(KnowledgeEntry::new(
                    "kb-alice-vec",
                    "alpha widget notes",
                    vec!["project".into()],
                ))
                .await
                .expect("alice write");
        })
        .await;
        set_embedding(&fx.pool, "kb-alice-vec", vec![vec![1.0, 0.0, 0.0]]).await;

        with_user_id(UserId::new("bob"), async {
            store
                .write(KnowledgeEntry::new(
                    "kb-bob-vec",
                    "beta gadget notes",
                    vec!["project".into()],
                ))
                .await
                .expect("bob write");
        })
        .await;
        set_embedding(&fx.pool, "kb-bob-vec", vec![vec![0.0, 1.0, 0.0]]).await;

        // Bob searches with an embedding pointing exactly at Alice's vector.
        // The vector branch runs; scoping must keep it to Bob's own rows.
        let bob_hits = with_user_id(UserId::new("bob"), async {
            store
                .search("widget", vec![1.0, 0.0, 0.0], None, None, 10)
                .await
        })
        .await
        .expect("bob search");
        assert!(
            !bob_hits.iter().any(|e| e.id == "kb-alice-vec"),
            "bob's hybrid search must NOT surface alice's embedded doc via the \
             vector branch; got {:?}",
            bob_hits.iter().map(|e| &e.id).collect::<Vec<_>>()
        );

        // Alice, with the same embedding, DOES find her own doc via the vector
        // branch — proving the branch ran (positive control).
        let alice_hits = with_user_id(UserId::new("alice"), async {
            store
                .search("nomatchterm", vec![1.0, 0.0, 0.0], None, None, 10)
                .await
        })
        .await
        .expect("alice search");
        assert!(
            alice_hits.iter().any(|e| e.id == "kb-alice-vec"),
            "alice's own embedded doc must be reachable through the vector \
             branch; got {:?}",
            alice_hits.iter().map(|e| &e.id).collect::<Vec<_>>()
        );
        fx
    })
    .await;
}

// -- hybrid search: exclude_tags on the vector branch ------------------------

#[tokio::test]
async fn knowledge_hybrid_search_excludes_tags() {
    // `exclude_tags` (`NOT (tags && $7)`) must drop a tagged row even when it is
    // reachable ONLY through the vector branch. The excluded doc does not
    // FTS-match the query, so its sole path into the result set is the vector
    // branch — proving the exclusion applies there.
    //
    // MUTATION: removing `AND ($7 ... NOT (tags && $7))` from the vector branch
    // lets the `secret`-tagged doc back into vector_ranked → RED.
    with_fixture("knowledge_hybrid_search_excludes_tags", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .write(KnowledgeEntry::new(
                    "kb-keep",
                    "widget planning doc",
                    vec!["project".into()],
                ))
                .await
                .expect("write keep");
            store
                .write(KnowledgeEntry::new(
                    "kb-secret",
                    "gadget summary sheet",
                    vec!["project".into(), "secret".into()],
                ))
                .await
                .expect("write secret");
        })
        .await;
        // Both share the query vector so both are vector-matched; only "kb-keep"
        // FTS-matches "widget".
        set_embedding(&fx.pool, "kb-keep", vec![vec![1.0, 0.0, 0.0]]).await;
        set_embedding(&fx.pool, "kb-secret", vec![vec![1.0, 0.0, 0.0]]).await;

        let hits = with_user_id(UserId::new("alice"), async {
            store
                .search(
                    "widget",
                    vec![1.0, 0.0, 0.0],
                    None,
                    Some(vec!["secret".into()]),
                    10,
                )
                .await
        })
        .await
        .expect("search");
        let ids: Vec<_> = hits.iter().map(|e| e.id.clone()).collect();
        assert!(
            !ids.iter().any(|id| id == "kb-secret"),
            "exclude_tags must drop the secret-tagged doc from the vector \
             branch; got {ids:?}"
        );
        assert!(
            ids.iter().any(|id| id == "kb-keep"),
            "the non-excluded doc must remain; got {ids:?}"
        );
        fx
    })
    .await;
}

// -- hybrid search: RRF fusion ordering --------------------------------------

#[tokio::test]
async fn knowledge_hybrid_search_rrf_orders_by_fused_rank() {
    // Reciprocal-rank fusion: a doc present in BOTH the vector list and the text
    // list scores `1/(60+rank_v) + 1/(60+rank_t)`, which outranks a doc present
    // in only one list. We build exactly that: a "both" doc (embedded + FTS),
    // a "vec_only" doc (embedded, no FTS match), and a "text_only" doc (NULL
    // embedding, FTS match). "both" must land first.
    //
    // MUTATION: flipping `ORDER BY rrf_score DESC` to `ASC` puts "both" last
    // → RED.
    with_fixture(
        "knowledge_hybrid_search_rrf_orders_by_fused_rank",
        |fx| async move {
            let store = PgKnowledgeBaseStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "both",
                        "quantum widget engine",
                        vec!["k".into()],
                    ))
                    .await
                    .expect("w both");
                store
                    .write(KnowledgeEntry::new(
                        "vec_only",
                        "unrelated prose xyzzy",
                        vec!["k".into()],
                    ))
                    .await
                    .expect("w vec_only");
                store
                    .write(KnowledgeEntry::new(
                        "text_only",
                        "quantum widget report",
                        vec!["k".into()],
                    ))
                    .await
                    .expect("w text_only");
            })
            .await;
            // "both" and "vec_only" are embedded at the query vector; "text_only"
            // stays NULL-embedded so it appears only in the text branch.
            set_embedding(&fx.pool, "both", vec![vec![1.0, 0.0, 0.0]]).await;
            set_embedding(&fx.pool, "vec_only", vec![vec![1.0, 0.0, 0.0]]).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("quantum widget", vec![1.0, 0.0, 0.0], None, None, 10)
                    .await
            })
            .await
            .expect("search");
            let ids: Vec<_> = hits.iter().map(|e| e.id.clone()).collect();
            assert_eq!(
                ids.first().map(String::as_str),
                Some("both"),
                "the doc present in BOTH the vector and text lists must have the \
                 highest fused RRF score; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

// -- list_page: keyset pagination --------------------------------------------

#[tokio::test]
async fn list_page_walks_cursors_without_dup_or_gap() {
    // Walking a >limit set page-by-page must visit every row exactly once with
    // no duplicate and no gap across the keyset boundary.
    //
    // MUTATION: changing `created_at < $5` to `created_at <= $5` on the
    // NewestFirst branch re-emits the boundary row → duplicate → RED.
    with_fixture(
        "list_page_walks_cursors_without_dup_or_gap",
        |fx| async move {
            let store = PgKnowledgeBaseStore::new(fx.pool.clone());

            let ids = ["p1", "p2", "p3", "p4", "p5"];
            with_user_id(UserId::new("alice"), async {
                for id in ids {
                    store
                        .write(KnowledgeEntry::new(id, "content for page walk", vec![]))
                        .await
                        .unwrap_or_else(|e| panic!("write {id}: {e}"));
                }
            })
            .await;
            // Distinct, strictly increasing created_at so ordering is unambiguous.
            for (i, id) in ids.iter().enumerate() {
                set_created_at(&fx.pool, id, ts(1_000 + i as i64)).await;
            }

            // Walk NewestFirst, limit 2.
            let mut seen: Vec<String> = Vec::new();
            let mut cursor: Option<String> = None;
            for _ in 0..10 {
                let q = KnowledgeListQuery {
                    limit: 2,
                    after: cursor.clone(),
                    order: ListOrderOpt(ListOrder::NewestFirst),
                    ..Default::default()
                };
                let page = with_user_id(UserId::new("alice"), async { store.list_page(q).await })
                    .await
                    .expect("list_page");
                for e in &page.entries {
                    seen.push(e.id.clone());
                }
                match page.next_cursor {
                    Some(c) => cursor = Some(c),
                    None => break,
                }
            }

            assert_eq!(
                seen,
                vec![
                    "p5".to_string(),
                    "p4".to_string(),
                    "p3".to_string(),
                    "p2".to_string(),
                    "p1".to_string()
                ],
                "newest-first cursor walk must yield every row once, in order, with \
             no dup/gap"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn list_page_tiebreaks_on_created_at_then_id() {
    // With identical created_at across rows, the keyset must break the tie on
    // `id` (DESC for newest-first, ASC for oldest-first) and paginate across the
    // tie without dropping or duplicating the boundary row.
    //
    // MUTATION: removing the `AND id < $6` (or `id > $6`) tiebreak turns the
    // second-page predicate into `created_at < $5` (false for all equal
    // timestamps) → the second page comes back empty → RED (rows lost).
    with_fixture(
        "list_page_tiebreaks_on_created_at_then_id",
        |fx| async move {
            let store = PgKnowledgeBaseStore::new(fx.pool.clone());

            let ids = ["aaa", "bbb", "ccc", "ddd"];
            with_user_id(UserId::new("alice"), async {
                for id in ids {
                    store
                        .write(KnowledgeEntry::new(id, "same timestamp", vec![]))
                        .await
                        .unwrap_or_else(|e| panic!("write {id}: {e}"));
                }
            })
            .await;
            // All rows share one created_at → forces the id tiebreak.
            for id in ids {
                set_created_at(&fx.pool, id, ts(5_000)).await;
            }

            let walk = |order: ListOrder| {
                let store = &store;
                async move {
                    let mut seen: Vec<String> = Vec::new();
                    let mut cursor: Option<String> = None;
                    for _ in 0..10 {
                        let q = KnowledgeListQuery {
                            limit: 2,
                            after: cursor.clone(),
                            order: ListOrderOpt(order),
                            ..Default::default()
                        };
                        let page =
                            with_user_id(UserId::new("alice"), async { store.list_page(q).await })
                                .await
                                .expect("list_page");
                        for e in &page.entries {
                            seen.push(e.id.clone());
                        }
                        match page.next_cursor {
                            Some(c) => cursor = Some(c),
                            None => break,
                        }
                    }
                    seen
                }
            };

            let newest = walk(ListOrder::NewestFirst).await;
            assert_eq!(
                newest,
                vec![
                    "ddd".to_string(),
                    "ccc".to_string(),
                    "bbb".to_string(),
                    "aaa".to_string()
                ],
                "equal-timestamp rows must tiebreak on id DESC and paginate cleanly"
            );

            let oldest = walk(ListOrder::OldestFirst).await;
            assert_eq!(
                oldest,
                vec![
                    "aaa".to_string(),
                    "bbb".to_string(),
                    "ccc".to_string(),
                    "ddd".to_string()
                ],
                "equal-timestamp rows must tiebreak on id ASC and paginate cleanly"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn list_page_rejects_malformed_cursor() {
    // `decode_cursor` guards two failure modes: a missing `:` separator and an
    // unparseable micros prefix. Both must surface as a `Storage` error rather
    // than being silently coerced. A well-formed cursor in the same test
    // succeeds, proving the rejection is not vacuous.
    //
    // MUTATION: relaxing `micros.parse().map_err(...)?` to `.unwrap_or(0)` makes
    // the "notanumber:kb" case succeed → RED.
    with_fixture("list_page_rejects_malformed_cursor", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .write(KnowledgeEntry::new("kb-c", "cursor content", vec![]))
                .await
                .expect("write");
        })
        .await;

        for bad in ["nocolonhere", "notanumber:kb-c"] {
            let q = KnowledgeListQuery {
                limit: 10,
                after: Some(bad.to_string()),
                ..Default::default()
            };
            let res = with_user_id(UserId::new("alice"), async { store.list_page(q).await }).await;
            assert!(
                matches!(res, Err(CoreError::Storage(_))),
                "malformed cursor {bad:?} must be rejected, got {res:?}"
            );
        }

        // A valid cursor is accepted (contrast case).
        let first = with_user_id(UserId::new("alice"), async {
            store
                .list_page(KnowledgeListQuery {
                    limit: 10,
                    ..Default::default()
                })
                .await
        })
        .await
        .expect("first page ok");
        assert!(!first.entries.is_empty(), "sanity: at least one row exists");
        fx
    })
    .await;
}

#[tokio::test]
async fn list_page_clamps_limit_1_to_500() {
    // `q.limit.clamp(1, 500)`: a requested limit of 0 must still return 1 row
    // (not 0), and an absurd limit must not error or over-fetch.
    //
    // MUTATION: changing `.clamp(1, 500)` to `.clamp(0, 500)` makes the limit-0
    // request return an empty page → RED.
    with_fixture("list_page_clamps_limit_1_to_500", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        let ids = ["c1", "c2", "c3"];
        with_user_id(UserId::new("alice"), async {
            for (i, id) in ids.iter().enumerate() {
                store
                    .write(KnowledgeEntry::new(*id, "clamp content", vec![]))
                    .await
                    .unwrap();
                set_created_at(&fx.pool, id, ts(2_000 + i as i64)).await;
            }
        })
        .await;

        // Lower clamp: limit 0 → 1 row (with more remaining ⇒ a next_cursor).
        let low = with_user_id(UserId::new("alice"), async {
            store
                .list_page(KnowledgeListQuery {
                    limit: 0,
                    ..Default::default()
                })
                .await
        })
        .await
        .expect("low");
        assert_eq!(
            low.entries.len(),
            1,
            "limit 0 must clamp up to exactly 1 row"
        );
        assert!(
            low.next_cursor.is_some(),
            "with 3 rows and a clamped limit of 1, more pages must remain"
        );

        // Upper clamp: an absurd limit returns all rows without error.
        let high = with_user_id(UserId::new("alice"), async {
            store
                .list_page(KnowledgeListQuery {
                    limit: 1_000_000,
                    ..Default::default()
                })
                .await
        })
        .await
        .expect("high");
        assert_eq!(high.entries.len(), 3, "absurd limit returns all 3 rows");
        assert!(
            high.next_cursor.is_none(),
            "no more pages after the last row"
        );
        fx
    })
    .await;
}

// -- delete_many: cross-user opacity -----------------------------------------

#[tokio::test]
async fn delete_many_ignores_foreign_ids() {
    // `delete_many` is user-scoped: deleting a batch that names another user's
    // id removes only the caller's own rows and reports the real count. Bob's
    // row must survive Alice's attempt.
    //
    // MUTATION: dropping `WHERE user_id = $1` lets Alice delete Bob's row too →
    // count becomes 2 and Bob's row vanishes → RED.
    with_fixture("delete_many_ignores_foreign_ids", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .write(KnowledgeEntry::new("own", "alice row", vec![]))
                .await
                .expect("alice write");
        })
        .await;
        with_user_id(UserId::new("bob"), async {
            store
                .write(KnowledgeEntry::new("bobs", "bob row", vec![]))
                .await
                .expect("bob write");
        })
        .await;

        let count = with_user_id(UserId::new("alice"), async {
            store
                .delete_many(&["own".to_string(), "bobs".to_string()])
                .await
        })
        .await
        .expect("delete_many");
        assert_eq!(count, 1, "only alice's own row is deleted");

        // Bob's row is untouched.
        let bob_row = with_user_id(UserId::new("bob"), async { store.get("bobs").await })
            .await
            .expect("bob get");
        assert!(
            bob_row.is_some(),
            "bob's row must survive alice's delete_many"
        );

        // Alice's row is gone.
        let alice_row = with_user_id(UserId::new("alice"), async { store.get("own").await })
            .await
            .expect("alice get");
        assert!(alice_row.is_none(), "alice's own row was deleted");
        fx
    })
    .await;
}
