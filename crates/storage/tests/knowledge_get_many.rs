//! The batch read behind `builtin_knowledge_base_get` (#1120).
//!
//! `PgKnowledgeBaseStore::get_many` is the only path that turns an id the model
//! already holds back into an entry. Two of its rules cannot be proved without
//! a real database, because both live in the statement's `WHERE` clause:
//!
//! - The read is scoped by `user_id`. Row-level security is a non-FORCE
//!   backstop that the table owner bypasses, so the predicate is the only real
//!   guard. Another user's id must be indistinguishable from an id that never
//!   existed - absent from the result, with nothing that says it exists.
//! - A retired (soft-deleted) entry is missing. Every other read filters
//!   `deleted_at IS NULL`, and a batch read that did not would hand back
//!   content the store has already withdrawn.
//!
//! ## Running locally
//!
//! ```sh
//! just test-db --test knowledge_get_many
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
use desktop_assistant_storage::{PgKnowledgeBaseStore, UserId, with_user_id};
use sqlx::PgPool;

const ALICE: &str = "kb-get-alice";
const BOB: &str = "kb-get-bob";

/// Boot a fixture in its own schema with migrations applied. `None` when
/// `TEST_DATABASE_URL` is unset, which is how each test pass-skips.
async fn fixture() -> Option<support::DbFixture> {
    support::DbFixture::try_new("kb1120").await
}

/// Write one entry as `user`.
async fn write_as(store: &PgKnowledgeBaseStore, user: &str, id: &str, content: &str) {
    with_user_id(UserId::new(user), async {
        store
            .write(KnowledgeEntry::new(id, content, vec!["memory".into()]))
            .await
            .unwrap_or_else(|e| panic!("write {id}: {e}"));
    })
    .await;
}

/// Retire a row directly, as a fixture setup step. Not a claim about which
/// production path produces a tombstone this way - consolidation itself no
/// longer soft-deletes anything (#893).
async fn soft_delete(pool: &PgPool, id: &str) {
    let res = sqlx::query("UPDATE knowledge_base SET deleted_at = NOW() WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .expect("soft delete");
    assert_eq!(res.rows_affected(), 1, "soft delete should touch row {id}");
}

/// The ids `get_many` resolved for `user`, sorted so the assertion does not
/// depend on the order Postgres happened to return rows in.
async fn get_ids_as(store: &PgKnowledgeBaseStore, user: &str, ids: &[&str]) -> Vec<String> {
    let ids: Vec<String> = ids.iter().map(|i| (*i).to_string()).collect();
    let mut found: Vec<String> = with_user_id(UserId::new(user), async {
        store.get_many(&ids).await.expect("get_many succeeds")
    })
    .await
    .into_iter()
    .map(|e| e.id)
    .collect();
    found.sort();
    found
}

#[tokio::test]
async fn kb_get_treats_another_tenants_id_as_missing() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

    // `knowledge_base.id` is a global primary key, so Bob can name Alice's id
    // exactly. He must get the same answer he would get for an id nobody holds.
    write_as(&store, ALICE, "kb-alice-only", "alice's durable fact").await;
    write_as(&store, BOB, "kb-bob-own", "bob's durable fact").await;

    let bobs = get_ids_as(&store, BOB, &["kb-bob-own", "kb-alice-only"]).await;
    assert_eq!(
        bobs,
        vec!["kb-bob-own".to_string()],
        "bob's batch read must resolve his own entry and miss alice's"
    );

    let invented = get_ids_as(&store, BOB, &["kb-nobody-holds-this"]).await;
    assert_eq!(
        invented,
        Vec::<String>::new(),
        "an id nobody holds resolves to nothing"
    );
    assert_eq!(
        get_ids_as(&store, BOB, &["kb-alice-only"]).await,
        invented,
        "another tenant's id must answer exactly as an id that never existed does"
    );

    // Alice still reads her own entry, so the miss above is scoping and not a
    // row that failed to land.
    assert_eq!(
        get_ids_as(&store, ALICE, &["kb-alice-only"]).await,
        vec!["kb-alice-only".to_string()]
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn kb_get_treats_a_retired_entry_as_missing() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

    write_as(&store, ALICE, "kb-live", "still true").await;
    write_as(&store, ALICE, "kb-retired", "superseded and withdrawn").await;
    soft_delete(&fx.pool, "kb-retired").await;

    let found = get_ids_as(&store, ALICE, &["kb-live", "kb-retired"]).await;
    assert_eq!(
        found,
        vec!["kb-live".to_string()],
        "a retired entry must not come back, and must not fail the batch"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn kb_get_treats_an_id_no_row_can_hold_as_missing() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

    // A stored id cannot contain a NUL byte, because Postgres `text` cannot
    // hold one. So an id carrying one names nothing, and must answer the way
    // every other id that names nothing answers. Sent to the database it does
    // not miss, it raises - and takes every other id in the batch with it.
    write_as(&store, ALICE, "kb-live", "still true").await;

    let found = get_ids_as(&store, ALICE, &["kb-live", "kb\u{0}broken"]).await;
    assert_eq!(
        found,
        vec!["kb-live".to_string()],
        "an id no row can hold is one miss, not a failed batch"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn kb_get_resolves_a_batch_in_one_read() {
    let Some(fx) = fixture().await else { return };
    let store = PgKnowledgeBaseStore::new(fx.pool.clone(), KnowledgeDeletePolicy::default());

    for i in 0..3 {
        write_as(&store, ALICE, &format!("kb-{i}"), &format!("fact {i}")).await;
    }

    let found = get_ids_as(&store, ALICE, &["kb-0", "kb-1", "kb-2", "kb-absent"]).await;
    assert_eq!(
        found,
        vec!["kb-0".to_string(), "kb-1".to_string(), "kb-2".to_string()],
        "every id the owner holds resolves; the one nobody holds is simply absent"
    );

    // An empty request is a successful empty answer, never a statement with no
    // parameters.
    let empty = with_user_id(UserId::new(ALICE), async {
        store.get_many(&[]).await.expect("empty get_many succeeds")
    })
    .await;
    assert!(empty.is_empty());

    fx.cleanup().await;
}
