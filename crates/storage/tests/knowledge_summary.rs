//! The knowledge base's one-line `summary` column (issue #1097).
//!
//! A knowledge entry had no short form: a reader that wanted to show many
//! entries at once printed the whole content or cut it at a byte count. The
//! `summary` column is that short form, and these tests pin the three things
//! the storage layer owes it.
//!
//! 1. **Every read path carries it.** `get`, `list`, `list_page`, `search_text`
//!    and the hybrid `search` all select the column. The hybrid query fuses two
//!    ranked sub-queries, so a column it forgets to carry through any one of
//!    the common table expressions is dropped silently.
//! 2. **A row that predates the column reads back as absent.** The column is
//!    nullable because hundreds of existing rows have no summary and the
//!    migration cannot invent one.
//! 3. **A write that says nothing about the summary preserves it.** This is the
//!    rule `source` already follows. Issue #1093 is the same defect on the tool
//!    path: a partial update wiped the fields it did not mention, and every
//!    tag-filtered search then lost the entry.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_summary
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::{
    KnowledgeBaseStore, KnowledgeListQuery, ListOrder, ListOrderOpt,
};
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, with_user_id};
use pgvector::Vector;
use sqlx::PgPool;

const USER: &str = "kb-summary-user";

/// The model every seeded embedding is stamped with, and the one every hybrid
/// search below passes: the vector arm only considers rows embedded by the
/// query's own model.
const MODEL: &str = "test-model";

/// Boot a fixture in its own schema with migrations applied. `None` when
/// `TEST_DATABASE_URL` is unset, which is how each test pass-skips.
async fn fixture() -> Option<support::DbFixture> {
    let fx = support::DbFixture::try_new("kb1097").await;
    if fx.is_none() {
        eprintln!("skip: TEST_DATABASE_URL not set");
    }
    fx
}

/// Build an entry carrying a summary.
fn entry_with_summary(id: &str, content: &str, summary: &str) -> KnowledgeEntry {
    let mut e = KnowledgeEntry::new(id, content, vec!["preference".into()]);
    e.summary = Some(summary.to_string());
    e
}

/// Stamp a `vector[]` embedding onto a row so the hybrid search's vector arm
/// actually runs (writes never embed inline; the background backfill does).
async fn set_embedding(pool: &PgPool, id: &str, chunk: Vec<f32>) {
    let vecs: Vec<Vector> = vec![Vector::from(chunk)];
    sqlx::query(
        "UPDATE knowledge_base \
         SET embedding = $1::vector[], embedding_model = $3, \
             embeddings_updated_at = NOW() \
         WHERE id = $2",
    )
    .bind(&vecs)
    .bind(id)
    .bind(MODEL)
    .execute(pool)
    .await
    .expect("stamp embedding");
}

// -- round trip --------------------------------------------------------------

#[tokio::test]
async fn knowledge_summary_round_trips_through_the_store() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        let saved = store
            .write(entry_with_summary(
                "kb-rt",
                "The user prefers a dark theme in every editor and terminal.",
                "Prefers dark themes",
            ))
            .await
            .expect("write succeeds");
        assert_eq!(
            saved.summary.as_deref(),
            Some("Prefers dark themes"),
            "the write reports back the summary it stored"
        );

        let read = store
            .get("kb-rt")
            .await
            .expect("get succeeds")
            .expect("the entry exists");
        assert_eq!(read.summary.as_deref(), Some("Prefers dark themes"));
    })
    .await;

    fx.cleanup().await;
}

// -- every read path carries it ---------------------------------------------

#[tokio::test]
async fn knowledge_list_carries_the_summary() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(entry_with_summary(
                "kb-list",
                "The user prefers a dark theme.",
                "Prefers dark themes",
            ))
            .await
            .expect("write succeeds");

        let entries = store.list(10, 0, None).await.expect("list succeeds");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].summary.as_deref(), Some("Prefers dark themes"));
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_list_page_carries_the_summary() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(entry_with_summary(
                "kb-page",
                "The user prefers a dark theme.",
                "Prefers dark themes",
            ))
            .await
            .expect("write succeeds");

        let page = store
            .list_page(KnowledgeListQuery {
                limit: 10,
                after: None,
                order: ListOrderOpt(ListOrder::NewestFirst),
                tags: None,
                exclude_tags: None,
                source: None,
            })
            .await
            .expect("list_page succeeds");
        assert_eq!(page.entries.len(), 1);
        assert_eq!(
            page.entries[0].summary.as_deref(),
            Some("Prefers dark themes")
        );
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_search_text_carries_the_summary() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(entry_with_summary(
                "kb-fts",
                "The user prefers a dark theme in the terminal.",
                "Prefers dark themes",
            ))
            .await
            .expect("write succeeds");

        let hits = store
            .search_text("terminal", None, 10)
            .await
            .expect("search_text succeeds");
        assert_eq!(hits.len(), 1, "the full-text arm found the seeded row");
        assert_eq!(hits[0].summary.as_deref(), Some("Prefers dark themes"));
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn knowledge_hybrid_search_carries_the_summary() {
    // The hybrid query fuses a vector-ranked and a text-ranked sub-query. A
    // column dropped from any one of those common table expressions vanishes
    // from the result without an error, so this drives the vector arm with a
    // real embedding rather than the full-text fallback.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(entry_with_summary(
                "kb-hybrid",
                "The user prefers a dark theme in the terminal.",
                "Prefers dark themes",
            ))
            .await
            .expect("write succeeds");
        set_embedding(&fx.pool, "kb-hybrid", vec![1.0, 0.0, 0.0]).await;

        let page = store
            .search("terminal", vec![1.0, 0.0, 0.0], MODEL, None, None, 10)
            .await
            .expect("search succeeds");
        assert_eq!(page.entries.len(), 1, "the hybrid search found the row");
        assert_eq!(
            page.entries[0].summary.as_deref(),
            Some("Prefers dark themes")
        );
    })
    .await;

    fx.cleanup().await;
}

// -- rows that predate the column -------------------------------------------

#[tokio::test]
async fn a_row_written_before_the_migration_reads_back_without_a_summary() {
    // Hundreds of rows existed before the column did. The migration adds it
    // nullable rather than inventing a value it cannot derive, so those rows
    // must read back as "no summary yet" and not fail the read.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    sqlx::query("INSERT INTO knowledge_base (id, user_id, content, tags, metadata) VALUES ($1, $2, $3, $4, $5)")
        .bind("kb-legacy")
        .bind(USER)
        .bind("A fact stored before summaries existed.")
        .bind(vec!["preference".to_string()])
        .bind(serde_json::json!({}))
        .execute(&fx.pool)
        .await
        .expect("seed a row with no summary");

    with_user_id(UserId::new(USER), async {
        let read = store
            .get("kb-legacy")
            .await
            .expect("get succeeds")
            .expect("the entry exists");
        assert_eq!(read.summary, None);
    })
    .await;

    fx.cleanup().await;
}

// -- a write that says nothing preserves it ----------------------------------

#[tokio::test]
async fn a_write_without_a_summary_leaves_the_stored_one_in_place() {
    // The rule `source` already follows: absent means "leave alone", never
    // "clear". Issue #1093 is this defect on the tool path - a partial update
    // wiped what it did not mention, and the entry fell out of every
    // tag-filtered search.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(entry_with_summary(
                "kb-preserve",
                "The user prefers a dark theme.",
                "Prefers dark themes",
            ))
            .await
            .expect("first write succeeds");

        // A second write that knows nothing about summaries - the shape every
        // caller has today, because nothing writes the field yet.
        let mut update = KnowledgeEntry::new(
            "kb-preserve",
            "The user prefers a dark theme everywhere.",
            vec!["preference".into()],
        );
        assert_eq!(update.summary, None, "premise: this write carries none");
        update.source = None;

        let saved = store.write(update).await.expect("second write succeeds");
        assert_eq!(
            saved.summary.as_deref(),
            Some("Prefers dark themes"),
            "a write that carries no summary must not clear the stored one"
        );

        let read = store
            .get("kb-preserve")
            .await
            .expect("get succeeds")
            .expect("the entry exists");
        assert_eq!(read.summary.as_deref(), Some("Prefers dark themes"));
        assert_eq!(read.content, "The user prefers a dark theme everywhere.");
    })
    .await;

    fx.cleanup().await;
}

#[tokio::test]
async fn a_write_that_carries_a_summary_replaces_the_stored_one() {
    // The other half of the preserve rule: an update that does state a summary
    // must land, or the dream cycle could never rewrite a stale one.
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone());

    with_user_id(UserId::new(USER), async {
        store
            .write(entry_with_summary(
                "kb-replace",
                "The user prefers a dark theme.",
                "Prefers dark themes",
            ))
            .await
            .expect("first write succeeds");

        let saved = store
            .write(entry_with_summary(
                "kb-replace",
                "The user prefers a light theme after all.",
                "Prefers light themes",
            ))
            .await
            .expect("second write succeeds");

        assert_eq!(saved.summary.as_deref(), Some("Prefers light themes"));
    })
    .await;

    fx.cleanup().await;
}
