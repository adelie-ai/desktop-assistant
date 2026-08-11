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

use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::situation::{
    FieldFan, Situation, SituationCue, SituationField,
};
use desktop_assistant_core::ports::knowledge::{
    KnowledgeBaseStore, KnowledgeListQuery, ListOrder, ListOrderOpt,
};
use desktop_assistant_core::ports::knowledge_use::{
    KnowledgeUseLog, OfferScope, with_situation_cue,
};
use desktop_assistant_storage::{
    PgKnowledgeBaseStore, PgKnowledgeUseLog, UserId, run_migrations, with_user_id,
};
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

/// The model every seeded row is stamped with, and the one every search below
/// passes: the vector arm only considers rows embedded by the query's own model.
const MODEL: &str = "test-model";

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

/// Force a row's `updated_at` so a fused-score tie also ties on the
/// `search_hybrid` ORDER BY's second column, leaving `id` as the only
/// remaining tiebreaker (#1107 follow-up).
async fn set_updated_at(pool: &PgPool, id: &str, ts: chrono::DateTime<chrono::Utc>) {
    sqlx::query("UPDATE knowledge_base SET updated_at = $1 WHERE id = $2")
        .bind(ts)
        .bind(id)
        .execute(pool)
        .await
        .expect("set updated_at");
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
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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
                .search("widget", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
                .await
        })
        .await
        .expect("bob search")
        .entries;
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
                .search("nomatchterm", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
                .await
        })
        .await
        .expect("alice search")
        .entries;
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
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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
                    MODEL,
                    None,
                    Some(vec!["secret".into()]),
                    10,
                )
                .await
        })
        .await
        .expect("search")
        .entries;
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

// -- hybrid search: activation ordering (#1167) ------------------------------

#[tokio::test]
async fn knowledge_hybrid_search_orders_by_activation_and_not_by_a_fused_rank() {
    // Acceptance (#1167): the page is ordered by the activation score, over the
    // store's own spread, and not by any fusion of the two arms' ranks.
    //
    // The fixture is built so the two rules disagree. "further" is embedded a
    // little away from the query vector AND carries the query's words, so
    // reciprocal-rank fusion scores it `1/(60+2) + 1/(60+1)` - the highest
    // score on the page. "nearest" is embedded exactly at the query vector and
    // carries none of the query's words, so fusion scores it `1/(60+1)` alone
    // and puts it second.
    //
    // Under activation the distance survives into the score instead of being
    // discarded for a position, so "nearest" leads. MUTATION: restoring the RRF
    // ordering flips this pair, which is what makes it a real red-to-green
    // reproduction rather than a pin.
    with_fixture(
        "knowledge_hybrid_search_orders_by_activation_and_not_by_a_fused_rank",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "further",
                        "quantum widget engine",
                        vec!["k".into()],
                    ))
                    .await
                    .expect("w further");
                store
                    .write(KnowledgeEntry::new(
                        "nearest",
                        "unrelated prose xyzzy",
                        vec!["k".into()],
                    ))
                    .await
                    .expect("w nearest");
            })
            .await;
            set_embedding(&fx.pool, "nearest", vec![vec![1.0, 0.0, 0.0]]).await;
            set_embedding(&fx.pool, "further", vec![vec![0.9, 0.44, 0.0]]).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("quantum widget", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("search")
            .entries;
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids,
                vec!["nearest", "further"],
                "the nearer row must lead, because the distance survives into the score \
                 instead of being discarded for a rank; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_hybrid_search_puts_a_row_it_cannot_compare_after_the_ones_it_measured() {
    // A row with no stored vector carries no distance, so there is no
    // dimensionless term for the score to add and no honest place for it among
    // the measured rows. It keeps the order the database ranked it in and
    // follows - which is what keeps an entry written since the last embedding
    // backfill reachable at all.
    with_fixture(
        "knowledge_hybrid_search_puts_a_row_it_cannot_compare_after_the_ones_it_measured",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "unembedded",
                        "quantum widget report",
                        vec![],
                    ))
                    .await
                    .expect("w unembedded");
                store
                    .write(KnowledgeEntry::new(
                        "embedded",
                        "quantum widget engine",
                        vec![],
                    ))
                    .await
                    .expect("w embedded");
            })
            .await;
            // "unembedded" is left NULL-embedded on purpose: it is the state
            // every entry is in between its write and the next backfill pass.
            set_embedding(&fx.pool, "embedded", vec![vec![1.0, 0.0, 0.0]]).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("quantum widget", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("search")
            .entries;
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids,
                vec!["embedded", "unembedded"],
                "a row nothing measured must still travel, after the rows that were measured; \
                 got {ids:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_hybrid_search_ranks_an_opened_entry_above_a_nearer_unopened_one() {
    // Acceptance (#1167): the reinforcement half of the activation score
    // reaches the search tool, so the tool and the [Recall] block rank by one
    // rule. This is the half no rank fusion can express at all: a position has
    // discarded both the distance and the use log by the time it is a position.
    //
    // "used" sits a little further from the query vector than "unread" and has
    // been offered and opened; "unread" has never been read. The gap in
    // distance is small enough that the log decides it, which is exactly the
    // near-tie the term exists to settle.
    with_fixture(
        "knowledge_hybrid_search_ranks_an_opened_entry_above_a_nearer_unopened_one",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            let log = PgKnowledgeUseLog::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new("used", "the deploy window", vec![]))
                    .await
                    .expect("w used");
                store
                    .write(KnowledgeEntry::new("unread", "the deploy window", vec![]))
                    .await
                    .expect("w unread");
            })
            .await;
            // Two hundredths of cosine distance apart, which the store's
            // stated estimate reads as about four tenths of a deviation - the
            // near tie the reinforcement term exists to settle, rather than a
            // gap so small any term at all would close it.
            set_embedding(&fx.pool, "unread", vec![vec![1.0, 0.0, 0.0]]).await;
            set_embedding(&fx.pool, "used", vec![vec![0.98, 0.199, 0.0]]).await;

            let ids = with_user_id(UserId::new("alice"), async {
                // Ten offers taken up: an entry the work keeps needing.
                for _ in 0..10 {
                    log.record_offered(OfferScope::recall("conv-1"), vec!["used".to_string()])
                        .await
                        .expect("record the offer");
                    log.record_opened(
                        "conv-1".to_string(),
                        vec!["used".to_string()],
                        Situation::new(),
                    )
                    .await
                    .expect("record the open");
                }
                let hits = store
                    .search(
                        "the deploy window",
                        vec![1.0, 0.0, 0.0],
                        MODEL,
                        None,
                        None,
                        10,
                    )
                    .await
                    .expect("search")
                    .entries;
                hits.into_iter().map(|e| e.id).collect::<Vec<_>>()
            })
            .await;

            assert_eq!(
                ids,
                vec!["used".to_string(), "unread".to_string()],
                "an entry the work keeps needing must lead a marginally nearer one nothing \
                 has opened; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

// -- the situation as a cue on the tool's own page (#1244) --------------------

/// The present situation the two tests below run in.
fn here_and_now() -> Situation {
    Situation::new().with(SituationField::Host, "workshop")
}

/// The cue a turn would have measured for the `[Recall]` block over a store
/// large enough to grade one: two hundred entries record the host, and a
/// quarter of them record this one.
///
/// Hand-built rather than measured over the fixture, because what these tests
/// are about is what the tool does with the cue the turn hands it - a
/// two-row fixture is below the population floor and would grade nothing.
fn a_turns_cue() -> SituationCue {
    let here = here_and_now();
    let fans = here
        .iter()
        .map(|(field, _)| {
            (
                field,
                FieldFan {
                    population: 200,
                    holding: 50,
                },
            )
        })
        .collect();
    SituationCue::measured(here, &fans).expect("two hundred entries is a gradeable store")
}

/// Seed two entries the query reaches identically, one of which has been seen
/// in the present situation.
///
/// Identical embeddings, identical text: the two tie on every term the score
/// reads except the situation, and the scan's own tie-break
/// (`ORDER BY distance, id DESC`) hands them over with the unsituated one
/// first. So an assertion that the situated one leads cannot pass by accident.
async fn seed_a_situated_pair(pool: &PgPool, store: &PgKnowledgeBaseStore) {
    let log = PgKnowledgeUseLog::new(pool.clone());
    with_user_id(UserId::new("alice"), async {
        store
            .write(KnowledgeEntry::new(
                "zz-elsewhere",
                "the deploy window",
                vec![],
            ))
            .await
            .expect("write zz-elsewhere");
        store
            .write(KnowledgeEntry::new("aa-here", "the deploy window", vec![]))
            .await
            .expect("write aa-here");
        log.record_situation(vec!["aa-here".to_string()], here_and_now())
            .await
            .expect("record the situation");
    })
    .await;
    set_embedding(pool, "zz-elsewhere", vec![vec![1.0, 0.0, 0.0]]).await;
    set_embedding(pool, "aa-here", vec![vec![1.0, 0.0, 0.0]]).await;
}

/// The ids one search answers with, run inside `cue`'s turn.
async fn search_ids(store: &PgKnowledgeBaseStore, cue: Option<SituationCue>) -> Vec<String> {
    with_user_id(UserId::new("alice"), async {
        with_situation_cue(cue, async {
            store
                .search(
                    "the deploy window",
                    vec![1.0, 0.0, 0.0],
                    MODEL,
                    None,
                    None,
                    10,
                )
                .await
                .expect("search")
                .entries
                .into_iter()
                .map(|e| e.id)
                .collect::<Vec<_>>()
        })
        .await
    })
    .await
}

#[tokio::test]
async fn a_search_inside_a_turn_ranks_an_entry_seen_in_the_present_situation_first() {
    // Acceptance (#1244): the situation term reaches the search tool's page.
    // The tool reads the cue the turn measured for the [Recall] block, and
    // reads each candidate's own situation record out of the log, so the two
    // paths rank one store by one rule.
    //
    // MUTATION: passing `None` for the cue in `search_hybrid`, or dropping the
    // situation read, turns this RED - the pair then comes back in the scan's
    // own order.
    with_fixture(
        "a_search_inside_a_turn_ranks_an_entry_seen_in_the_present_situation_first",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed_a_situated_pair(&fx.pool, &store).await;

            let ids = search_ids(&store, Some(a_turns_cue())).await;

            assert_eq!(
                ids,
                vec!["aa-here".to_string(), "zz-elsewhere".to_string()],
                "an entry this situation recurs with must lead an equally similar one it does \
                 not; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn a_search_outside_any_turn_ranks_the_page_as_it_ranked_before_the_cue() {
    // Acceptance (#1244): a turn with nothing connected, and any caller outside
    // a turn at all, installs no cue - and the page is then the page this tool
    // answered before the term reached it, over the very same stored situation
    // records.
    with_fixture(
        "a_search_outside_any_turn_ranks_the_page_as_it_ranked_before_the_cue",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed_a_situated_pair(&fx.pool, &store).await;

            let ids = search_ids(&store, None).await;

            assert_eq!(
                ids,
                vec!["zz-elsewhere".to_string(), "aa-here".to_string()],
                "with no cue the page must keep the order the scan gave it; got {ids:?}"
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
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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
        let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

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

// -- write path: tags are normalized (case/whitespace/facet, dedup) ----------

#[tokio::test]
async fn knowledge_write_normalizes_tags_case_whitespace_and_preserves_facets() {
    // Exact-match tag filters (`tags && $2`) fragment when the same intent is
    // written as `Preference`/`preference `/`preference`. The write path must
    // collapse that drift AND preserve a `facet:value` colon, then dedup — so a
    // round-trip through the store returns canonical, deduped tags.
    //
    // MUTATION: dropping the `normalize_tags(...)` call in `knowledge::write`
    // persists the raw tags verbatim → this test goes RED.
    with_fixture(
        "knowledge_write_normalizes_tags_case_whitespace_and_preserves_facets",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "kb-tag-norm",
                        "deploy notes for the adelie stack",
                        vec![
                            "Preference".into(),
                            " Memory ".into(),
                            "project:Adelie-AI".into(),
                            "preference".into(),
                        ],
                    ))
                    .await
                    .expect("write with drifty tags");
            })
            .await;

            let row = with_user_id(UserId::new("alice"), async {
                store.get("kb-tag-norm").await
            })
            .await
            .expect("get")
            .expect("row exists");

            // Case/whitespace collapsed, the duplicate "preference" dropped, and
            // the facet colon preserved (NOT mangled to `project-adelie-ai`).
            assert_eq!(
                row.tags,
                vec![
                    "preference".to_string(),
                    "memory".to_string(),
                    "project:adelie-ai".to_string(),
                ],
                "write must normalize + dedup tags and keep the facet colon"
            );
            fx
        },
    )
    .await;
}

// -- read path: tag filters are case/facet-insensitive (write/read symmetry) --

#[tokio::test]
async fn knowledge_read_filters_are_case_insensitive_and_symmetric() {
    // Writes normalize stored tags; reads must normalize the filter too, or a
    // differently-cased filter silently misses. Two docs share the FTS term
    // "deploy", so only the tag filter discriminates. Every read binding site is
    // exercised: the FTS include ($2) / exclude ($5), the vector-branch include
    // ($2), `list` ($1), and `list_page` include ($2) / exclude ($3).
    //
    // MUTATION: dropping `normalize_tag_filter(...)` on any read path makes the
    // differently-cased filter miss/keep the wrong rows → RED.
    with_fixture(
        "knowledge_read_filters_are_case_insensitive_and_symmetric",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "kb-instr",
                        "deploy runbook for the stack",
                        vec!["Instruction".into(), "project:Adelie-AI".into()],
                    ))
                    .await
                    .expect("write kb-instr");
                store
                    .write(KnowledgeEntry::new(
                        "kb-other",
                        "deploy runbook for something else",
                        vec!["memory".into(), "project:other".into()],
                    ))
                    .await
                    .expect("write kb-other");
            })
            .await;

            // INCLUDE via search_text (FTS $2): a differently-cased facet filter
            // still finds the matching row and only that row.
            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search_text("deploy", Some(vec!["Project:adelie-ai".into()]), 10)
                    .await
            })
            .await
            .expect("search_text include");
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert!(
                ids.contains(&"kb-instr") && !ids.contains(&"kb-other"),
                "case-varied include filter must find only kb-instr; got {ids:?}"
            );

            // EXCLUDE via the search FTS-fallback (empty embedding, $5): a
            // differently-cased exclude filter drops the matching row, keeps rest.
            let kept = with_user_id(UserId::new("alice"), async {
                store
                    .search(
                        "deploy",
                        vec![],
                        MODEL,
                        None,
                        Some(vec!["PROJECT:Adelie-AI".into()]),
                        10,
                    )
                    .await
            })
            .await
            .expect("search exclude")
            .entries;
            let kept_ids: Vec<&str> = kept.iter().map(|e| e.id.as_str()).collect();
            assert!(
                kept_ids.contains(&"kb-other") && !kept_ids.contains(&"kb-instr"),
                "case-varied exclude filter must drop kb-instr, keep kb-other; got {kept_ids:?}"
            );

            // INCLUDE on the hybrid VECTOR branch ($2): stamp embeddings so the
            // vector branch actually runs, then filter with a case-varied facet.
            set_embedding(&fx.pool, "kb-instr", vec![vec![1.0, 0.0, 0.0]]).await;
            set_embedding(&fx.pool, "kb-other", vec![vec![0.0, 1.0, 0.0]]).await;
            let vec_hits = with_user_id(UserId::new("alice"), async {
                store
                    .search(
                        "deploy",
                        vec![1.0, 0.0, 0.0],
                        MODEL,
                        Some(vec!["Project:Adelie-AI".into()]),
                        None,
                        10,
                    )
                    .await
            })
            .await
            .expect("vector search include")
            .entries;
            let vec_ids: Vec<&str> = vec_hits.iter().map(|e| e.id.as_str()).collect();
            assert!(
                vec_ids.contains(&"kb-instr") && !vec_ids.contains(&"kb-other"),
                "hybrid vector-branch include filter must find only kb-instr; got {vec_ids:?}"
            );

            // INCLUDE via list ($1).
            let via_list = with_user_id(UserId::new("alice"), async {
                store
                    .list(50, 0, Some(vec!["Project:Adelie-AI".into()]))
                    .await
            })
            .await
            .expect("list include");
            let list_ids: Vec<&str> = via_list.iter().map(|e| e.id.as_str()).collect();
            assert!(
                list_ids.contains(&"kb-instr") && !list_ids.contains(&"kb-other"),
                "list include filter must find only kb-instr; got {list_ids:?}"
            );

            // INCLUDE + EXCLUDE via list_page ($2 / $3), no FTS path.
            let inc = with_user_id(UserId::new("alice"), async {
                store
                    .list_page(KnowledgeListQuery {
                        limit: 50,
                        tags: Some(vec!["PROJECT:Adelie-AI".into()]),
                        ..Default::default()
                    })
                    .await
            })
            .await
            .expect("list_page include");
            let inc_ids: Vec<&str> = inc.entries.iter().map(|e| e.id.as_str()).collect();
            assert!(
                inc_ids.contains(&"kb-instr") && !inc_ids.contains(&"kb-other"),
                "list_page include filter must find only kb-instr; got {inc_ids:?}"
            );

            let exc = with_user_id(UserId::new("alice"), async {
                store
                    .list_page(KnowledgeListQuery {
                        limit: 50,
                        exclude_tags: Some(vec!["project:ADELIE-ai".into()]),
                        ..Default::default()
                    })
                    .await
            })
            .await
            .expect("list_page exclude");
            let exc_ids: Vec<&str> = exc.entries.iter().map(|e| e.id.as_str()).collect();
            assert!(
                exc_ids.contains(&"kb-other") && !exc_ids.contains(&"kb-instr"),
                "list_page exclude filter must drop kb-instr, keep kb-other; got {exc_ids:?}"
            );
            fx
        },
    )
    .await;
}

// -- #1107: the vector and text arms of `search_hybrid` must truncate to a
// -- DEFINED set (the nearest / highest-ranked candidates), not an arbitrary
// -- subset of the same size. `ROW_NUMBER() OVER (ORDER BY ...)` orders the
// -- window computation, not the statement's output; a `LIMIT` with no
// -- statement-level `ORDER BY` truncates a set the SQL standard never
// -- promises an order for.
//
// Both tests below seed more rows than `fetch_limit` (`limit * 2`), so the
// `LIMIT` in each CTE has something to actually cut.

#[tokio::test]
async fn vector_arm_truncates_to_the_nearest_candidates_not_an_arbitrary_subset() {
    // Contract PIN, not a red-to-green reproduction -- said plainly, per
    // #1107: this test passes both BEFORE and AFTER the `ORDER BY min_distance`
    // fix on PostgreSQL 17 today.
    //
    // Why it cannot be made to fail against the current query: `rank_v` feeds
    // `fused.rrf_score`, which the outer `ORDER BY rrf_score DESC` reads. A
    // column a later node reads is never eliminated, so the planner keeps
    // `Limit -> WindowAgg -> Sort(min_distance)` for `vector_ranked` -- proven
    // by `EXPLAIN (ANALYZE, VERBOSE)` against this exact scenario (20 rows,
    // `fetch_limit` 12): the candidate set IS the nearest 12, in rank order.
    // That plan shape is a property of how the result is consumed today, not
    // a guarantee the query states -- which is the whole defect. This test
    // guards the property going forward: it goes red the moment a refactor
    // stops reading `rank_v` downstream (or the planner changes), because at
    // that point the truncation reverts to a genuinely arbitrary subset.
    //
    // The FTS query term ("zzznomatchzzz") matches none of the seeded content,
    // so `text_ranked` is empty and `fused` reduces to `vector_ranked` alone --
    // isolating the vector arm from the fusion.
    with_fixture(
        "vector_arm_truncates_to_the_nearest_candidates_not_an_arbitrary_subset",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            // 20 rows -- more than fetch_limit (limit=6 -> fetch_limit=12).
            with_user_id(UserId::new("alice"), async {
                for i in 0..20u32 {
                    store
                        .write(KnowledgeEntry::new(
                            format!("row{i:02}"),
                            format!("distinct filler content row {i}"),
                            vec![],
                        ))
                        .await
                        .unwrap_or_else(|e| panic!("write row{i:02}: {e}"));
                }
            })
            .await;

            // Embeddings at strictly increasing cosine distance from [1,0,0]:
            // row00 is nearest, row19 is farthest.
            for i in 0..20u32 {
                let f = i as f32 * 0.01;
                set_embedding(&fx.pool, &format!("row{i:02}"), vec![vec![1.0 - f, f, 0.0]]).await;
            }

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("zzznomatchzzz", vec![1.0, 0.0, 0.0], MODEL, None, None, 6)
                    .await
            })
            .await
            .expect("search")
            .entries;
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids,
                vec!["row00", "row01", "row02", "row03", "row04", "row05"],
                "the vector arm's contribution must be exactly the 6 nearest \
                 rows, nearest first, out of 20 candidates and a fetch_limit \
                 of 12; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn text_arm_truncates_to_the_highest_ranked() {
    // The lexical arm's `ORDER BY ts_rank_cd(...) DESC` already sits before its
    // `LIMIT` in `text_ranked` -- this test pins that property rather than
    // proving a fix, matching #1107's acceptance criterion for the lexical arm.
    //
    // No row carries an embedding, so `chunk_distances` (which requires
    // `embedding IS NOT NULL`) is empty and `fused` reduces to `text_ranked`
    // alone -- isolating the text arm from the fusion.
    //
    // Each row's content repeats "wobble" a different number of times, at a
    // constant total word count, so `ts_rank_cd` (term-frequency weighted)
    // ranks row00 highest and row19 lowest, strictly monotonically.
    with_fixture(
        "text_arm_truncates_to_the_highest_ranked",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            // 20 rows -- more than fetch_limit (limit=6 -> fetch_limit=12).
            with_user_id(UserId::new("alice"), async {
                for i in 0..20u32 {
                    let content = format!(
                        "{}{}",
                        "wobble ".repeat((20 - i) as usize),
                        "pad ".repeat(i as usize)
                    );
                    store
                        .write(KnowledgeEntry::new(format!("row{i:02}"), content, vec![]))
                        .await
                        .unwrap_or_else(|e| panic!("write row{i:02}: {e}"));
                }
            })
            .await;

            // A non-empty query_embedding routes through search_hybrid rather
            // than the FTS-only fallback, even though no row is embedded.
            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("wobble", vec![1.0, 0.0, 0.0], MODEL, None, None, 6)
                    .await
            })
            .await
            .expect("search")
            .entries;
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids,
                vec!["row00", "row01", "row02", "row03", "row04", "row05"],
                "the text arm's contribution must be exactly the 6 \
                 highest-ranked rows, highest first, out of 20 candidates and \
                 a fetch_limit of 12; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

// -- #1107 follow-up: RRF generates exact ties by construction (a row found
// -- ONLY by one arm at rank 1 scores exactly `1/(60+1)`, whichever arm found
// -- it), so `fused`'s `ORDER BY rrf_score DESC` alone leaves the truncation
// -- undefined the same way a bare window `LIMIT` does. Unlike the
// -- window-function sites, a tie here is directly constructible, so this
// -- test is a real red-to-green reproduction, not a pin.

#[tokio::test]
async fn fused_search_truncates_to_a_defined_row_when_rrf_scores_tie() {
    // "tie-a" is found ONLY by the text arm (it is the sole row containing
    // the FTS query term "gronk", so it is rank_t = 1; it carries no
    // embedding, so `chunk_distances` -- which requires `embedding IS NOT
    // NULL` -- excludes it). "tie-b" is found ONLY by the vector arm (its
    // embedding exactly matches the query vector, so min_distance = 0 and it
    // is the sole, rank-1 member of `vector_ranked`; its content never
    // matches the FTS query term).
    //
    // Both therefore score exactly `1.0 / (60 + 1)`: one term is the literal
    // computation, the other is `COALESCE(..., 0)`. Same IEEE-754 formula,
    // same operands -- an exact tie, not an approximate one.
    //
    // Both rows' `updated_at` is forced equal, so the second ORDER BY column
    // also ties, leaving `id DESC` as the only column that can still decide
    // the winner. "tie-b" > "tie-a" lexicographically, so `id DESC` must pick
    // "tie-b" when `limit` truncates the tied pair down to 1.
    //
    // This assignment (not the reverse) is what makes the test a genuine
    // red-to-green: probed directly, the pre-fix query's `FULL OUTER JOIN`
    // consistently emits the text-arm-only row before the vector-arm-only row
    // when their scores tie, regardless of which literal id carries which
    // role. Naming the text-matched row "tie-a" (the lexicographically
    // SMALLER id) makes that incidental behavior disagree with `id DESC`,
    // so the assertion below fails before this PR's tiebreak and passes
    // after it.
    with_fixture(
        "fused_search_truncates_to_a_defined_row_when_rrf_scores_tie",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "tie-a",
                        "the distinctive gronk term appears in this content",
                        vec![],
                    ))
                    .await
                    .expect("write tie-a");
                store
                    .write(KnowledgeEntry::new(
                        "tie-b",
                        "unrelated prose that never matches the query text",
                        vec![],
                    ))
                    .await
                    .expect("write tie-b");
            })
            .await;
            set_embedding(&fx.pool, "tie-b", vec![vec![1.0, 0.0, 0.0]]).await;
            // tie-a is left unembedded on purpose.

            let same_instant = ts(9_000);
            set_updated_at(&fx.pool, "tie-a", same_instant).await;
            set_updated_at(&fx.pool, "tie-b", same_instant).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("gronk", vec![1.0, 0.0, 0.0], MODEL, None, None, 1)
                    .await
            })
            .await
            .expect("search")
            .entries;
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids,
                vec!["tie-b"],
                "with an exact rrf_score tie and an equal updated_at, the \
                 truncation to 1 row must be decided by id DESC (\"tie-b\" > \
                 \"tie-a\"), not by an undefined physical row order; got \
                 {ids:?}"
            );
            fx
        },
    )
    .await;
}

// -- hybrid search: the spread it measures (#1167) ----------------------------

/// The three statistics `HYBRID_SEARCH_SQL` computes, for the one query below.
#[derive(sqlx::FromRow)]
struct MeasuredSpread {
    median: Option<f64>,
    rows_read: i64,
    deviation: Option<f64>,
}

/// Bind the search scan and read back what it says the store's spread is.
///
/// `search_hybrid` consumes those three numbers and answers with entries, so
/// nothing downstream can tell a median that is wrong from one that is merely
/// absent - and a wrong spread silently changes every near tie the
/// reinforcement term exists to settle. This is why the query is held as its
/// own public string.
async fn measured_spread(pool: &PgPool, user: &str) -> MeasuredSpread {
    with_user_id(UserId::new(user), async {
        sqlx::query_as(desktop_assistant_storage::knowledge_search::HYBRID_SEARCH_SQL)
            .bind(Vector::from(vec![1.0_f32, 0.0, 0.0]))
            .bind(None::<Vec<String>>)
            .bind(100_i64)
            .bind("zzznomatchzzz")
            .bind(50_i64)
            .bind(user)
            .bind(None::<Vec<String>>)
            .bind(MODEL)
            .fetch_one(pool)
            .await
            .expect("the scan answers")
    })
    .await
}

#[tokio::test]
async fn the_hybrid_scan_measures_the_stores_own_median_and_deviation() {
    // Acceptance (#1167): the semantic term is the distance read against the
    // store's own spread, so the spread has to be the store's real one. The
    // seeded rows sit at known cosine distances - `1 - cos(angle)` for a unit
    // vector `angle` radians off the query - so both statistics are arithmetic
    // a reader can check by hand rather than numbers only the database knows.
    //
    // Twenty rows, evenly spread in angle, is also the smallest sample
    // `RecallDispersion::measured` will trust.
    with_fixture(
        "the_hybrid_scan_measures_the_stores_own_median_and_deviation",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            let mut distances: Vec<f64> = Vec::new();
            with_user_id(UserId::new("alice"), async {
                for i in 0..20u32 {
                    store
                        .write(KnowledgeEntry::new(
                            format!("row{i:02}"),
                            "a stored fact",
                            vec![],
                        ))
                        .await
                        .unwrap_or_else(|e| panic!("write row{i:02}: {e}"));
                }
            })
            .await;
            for i in 0..20u32 {
                let angle = (i + 1) as f32 * 0.1;
                set_embedding(
                    &fx.pool,
                    &format!("row{i:02}"),
                    vec![vec![angle.cos(), angle.sin(), 0.0]],
                )
                .await;
                distances.push(1.0 - f64::from(angle.cos()));
            }

            let measured = measured_spread(&fx.pool, "alice").await;

            distances.sort_by(f64::total_cmp);
            let expected_median = (distances[9] + distances[10]) / 2.0;
            let mut spread: Vec<f64> = distances
                .iter()
                .map(|d| (d - expected_median).abs())
                .collect();
            spread.sort_by(f64::total_cmp);
            let expected_deviation = (spread[9] + spread[10]) / 2.0;

            assert_eq!(measured.rows_read, 20, "every comparable row is measured");
            let median = measured
                .median
                .expect("a store of this size states a median");
            let deviation = measured
                .deviation
                .expect("a store of this size states a deviation");
            assert!(
                (median - expected_median).abs() < 1e-6,
                "the scan reported a median of {median} where the seeded distances give \
                 {expected_median}"
            );
            assert!(
                (deviation - expected_deviation).abs() < 1e-6,
                "the scan reported a deviation of {deviation} where the seeded distances give \
                 {expected_deviation}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn the_hybrid_scan_still_answers_when_no_row_can_be_compared() {
    // The construction that makes this work is the one place this scan differs
    // from the recall scan: its `s` selects from `m` rather than cross-joining
    // `d`, so it yields a row - with both statistics NULL - even when nothing
    // is embedded. Cross-joining an empty `d` would annihilate the whole
    // result, and the full-text arm's rows would vanish with it, which is the
    // ordinary state of a store between a write and the next backfill pass.
    with_fixture(
        "the_hybrid_scan_still_answers_when_no_row_can_be_compared",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new("only", "quantum widget report", vec![]))
                    .await
                    .expect("write only");
            })
            .await;
            // No embedding is stamped: nothing in this store can be compared.

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("quantum widget", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
                    .await
            })
            .await
            .expect("search")
            .entries;

            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids,
                vec!["only"],
                "a store with nothing to compare must still answer from its full-text arm"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_hybrid_search_admits_a_row_the_query_names_even_when_the_vector_arm_ranks_it_low()
 {
    // The defect this arm's admission exists to avoid, and the one shape of it
    // that no later ranking term could ever repair: a row the query names
    // exactly, which the vector arm CAN compare but ranks in the middle of the
    // store, must still reach the candidate set. Excluding every row the vector
    // arm reached - which is what "the full-text arm covers what the vector arm
    // cannot" reduces to on an embedded store - made the arm return nothing at
    // all, so no weight on any term could have lifted such a row back.
    //
    // MUTATION: adding `AND NOT EXISTS (SELECT 1 FROM d WHERE d.id = kb.id)`
    // back to the lexical arm drops "serial" from the answer entirely -> RED.
    //
    // The page ORDER is deliberately not asserted. On an embedded store this
    // row is ranked by its distance and sinks; #1239 is the term that would
    // lift it, and this test is what says the term will have something to lift.
    with_fixture(
        "knowledge_hybrid_search_admits_a_row_the_query_names_even_when_the_vector_arm_ranks_it_low",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                store
                    .write(KnowledgeEntry::new(
                        "serial",
                        "the widget carries serial gronk48219",
                        vec![],
                    ))
                    .await
                    .expect("write serial");
                for i in 0..8u32 {
                    store
                        .write(KnowledgeEntry::new(
                            format!("filler{i:02}"),
                            format!("unrelated prose about other matters {i}"),
                            vec![],
                        ))
                        .await
                        .unwrap_or_else(|e| panic!("write filler{i:02}: {e}"));
                }
            })
            .await;
            // Every filler row sits nearer the query vector than "serial" does,
            // so the vector arm ranks "serial" last of the nine.
            for i in 0..8u32 {
                let f = i as f32 * 0.01;
                set_embedding(
                    &fx.pool,
                    &format!("filler{i:02}"),
                    vec![vec![1.0 - f, f, 0.0]],
                )
                .await;
            }
            set_embedding(&fx.pool, "serial", vec![vec![0.0, 1.0, 0.0]]).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("gronk48219", vec![1.0, 0.0, 0.0], MODEL, None, None, 20)
                    .await
            })
            .await
            .expect("search")
            .entries;
            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert!(
                ids.contains(&"serial"),
                "a row the query names exactly must reach the candidate set however the vector \
                 arm ranks it; got {ids:?}"
            );
            assert_eq!(
                ids.iter().filter(|id| **id == "serial").count(),
                1,
                "both arms admit it, and the page must carry it once; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_hybrid_search_carries_provenance_so_salience_reads_the_same_as_the_block() {
    // The salience term reads an entry's own provenance, so a page that dropped
    // the column scored every deliberately-written entry below what the
    // [Recall] block scores it - the drift this work exists to remove, in the
    // one field the search projection did not carry.
    //
    // MUTATION: dropping `kb.source` from the projection, or restoring
    // `source: None` in `KbSearchRow::into_entry`, turns this RED.
    with_fixture(
        "knowledge_hybrid_search_carries_provenance_so_salience_reads_the_same_as_the_block",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

            with_user_id(UserId::new("alice"), async {
                let mut entry = KnowledgeEntry::new("deliberate", "the deploy window", vec![]);
                entry.source = Some("explicit".to_string());
                store.write(entry).await.expect("write deliberate");
            })
            .await;
            set_embedding(&fx.pool, "deliberate", vec![vec![1.0, 0.0, 0.0]]).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search(
                        "the deploy window",
                        vec![1.0, 0.0, 0.0],
                        MODEL,
                        None,
                        None,
                        10,
                    )
                    .await
            })
            .await
            .expect("search")
            .entries;

            assert_eq!(
                hits.first().and_then(|e| e.source.as_deref()),
                Some("explicit"),
                "the search page must carry the provenance the salience term reads"
            );
            fx
        },
    )
    .await;
}

// -- the query's own words (#1239) --------------------------------------------

/// Seed the store the identifier measurement was taken over: thirty rows the
/// query's words never reach, spread evenly in angle so the store states a real
/// dispersion, plus one row carrying a distinctive token and embedded so that
/// twelve of the thirty sit nearer to the query vector than it does.
///
/// Twelve nearer rows puts it at vector rank thirteen, which is where the
/// measurement in #1167's review found it.
async fn seed_the_identifier_store(pool: &PgPool, store: &PgKnowledgeBaseStore) {
    with_user_id(UserId::new("alice"), async {
        for i in 0..30u32 {
            store
                .write(KnowledgeEntry::new(
                    format!("filler{i:02}"),
                    format!("unrelated prose about other matters number {i}"),
                    vec![],
                ))
                .await
                .unwrap_or_else(|e| panic!("write filler{i:02}: {e}"));
        }
        store
            .write(KnowledgeEntry::new(
                "serial",
                "the widget shipped under serial gronk48219",
                vec![],
            ))
            .await
            .expect("write serial");
    })
    .await;

    // Angles spread from 0.02 to 0.60 radians off the query vector, so the
    // store has a real distribution rather than two clumps.
    for i in 0..30u32 {
        let angle = 0.02 + (i as f32) * 0.02;
        set_embedding(
            pool,
            &format!("filler{i:02}"),
            vec![vec![angle.cos(), angle.sin(), 0.0]],
        )
        .await;
    }
    // Twelve fillers sit at angles below 0.26; "serial" sits at 0.26, so twelve
    // rows are nearer and it is the thirteenth by distance.
    set_embedding(
        pool,
        "serial",
        vec![vec![0.26_f32.cos(), 0.26_f32.sin(), 0.0]],
    )
    .await;
}

#[tokio::test]
async fn knowledge_hybrid_search_puts_an_exactly_named_row_at_the_top_of_the_page() {
    // Acceptance (#1239): the measurement that blocked #1167, turned into a
    // test rather than left as a claim in a report.
    //
    // Before: "serial" sat thirteenth by cosine distance, the page took the
    // five nearest, and the row the query names exactly did not appear at all -
    // so the only text search the model has could not find an identifier.
    //
    // After: the words it carries are worth the spread this store's own
    // distances have, so it leads a page of five.
    //
    // MUTATION: returning `LexicalMatch::NONE` from `rank_page`, or dropping
    // `a.lexical_share` from the scan's projection, puts "serial" back off the
    // page -> RED.
    with_fixture(
        "knowledge_hybrid_search_puts_an_exactly_named_row_at_the_top_of_the_page",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed_the_identifier_store(&fx.pool, &store).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("gronk48219", vec![1.0, 0.0, 0.0], MODEL, None, None, 5)
                    .await
            })
            .await
            .expect("search")
            .entries;

            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert_eq!(
                ids.first().copied(),
                Some("serial"),
                "a row the query names exactly must lead a page of five, not sit thirteenth \
                 by distance; got {ids:?}"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_hybrid_search_leaves_a_row_the_query_never_names_where_its_distance_puts_it() {
    // Acceptance (#1239), the negative: a row with neither a text hit nor a
    // good distance must not be lifted. Over the same store, a query whose
    // words reach nothing leaves every row ranked on distance alone, so
    // "serial" stays thirteenth and off a page of five.
    //
    // This is what says the term reads the query's words rather than merely
    // reordering the page.
    with_fixture(
        "knowledge_hybrid_search_leaves_a_row_the_query_never_names_where_its_distance_puts_it",
        |fx| async move {
            let store =
                PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());
            seed_the_identifier_store(&fx.pool, &store).await;

            let hits = with_user_id(UserId::new("alice"), async {
                store
                    .search("zzznomatchzzz", vec![1.0, 0.0, 0.0], MODEL, None, None, 5)
                    .await
            })
            .await
            .expect("search")
            .entries;

            let ids: Vec<&str> = hits.iter().map(|e| e.id.as_str()).collect();
            assert!(
                !ids.contains(&"serial"),
                "with nothing for the query's words to reach, the term must lift nobody; \
                 got {ids:?}"
            );
            assert_eq!(
                ids,
                vec!["filler00", "filler01", "filler02", "filler03", "filler04"],
                "and the page is the five nearest, in order; got {ids:?}"
            );
            fx
        },
    )
    .await;
}
