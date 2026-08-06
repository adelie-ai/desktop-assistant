//! Integration tests for the per-conversation scratchpad store (#184).
//!
//! Exercises `PgScratchpadStore` end-to-end against a real Postgres with the
//! migration applied, proving batch upsert by key, `get_many`, ordered/limited
//! listing, FTS search, `delete_many`/`clear` counts, cascade-delete with the
//! parent conversation, and cross-user isolation.
//!
//! ## Running locally
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test scratchpad
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log line so
//! the suite stays green without a DB.

mod support;

use std::sync::Arc;

use std::collections::HashSet;

use desktop_assistant_core::domain::{Conversation, ConversationId, Message, Role};
use desktop_assistant_core::ports::scratchpad::{NewScratchpadNote, ScratchpadStore};
use desktop_assistant_core::ports::scratchpad_scope::{SubagentScope, with_subagent_scope};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{
    PgConversationStore, PgScratchpadStore, UserId, run_migrations, with_user_id,
};
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
        let schema = format!("issue184_{}", Uuid::now_v7().simple());

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

fn make_conversation(id: &str) -> Conversation {
    let mut conv = Conversation::new(id, "scratchpad test");
    conv.created_at = "2026-06-03 00:00:00".to_string();
    conv.updated_at = "2026-06-03 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

/// A plain (`note`-typed, unsequenced) upsert.
fn note(key: &str, content: &str) -> NewScratchpadNote {
    NewScratchpadNote::new(key, content)
}

/// A `todo`-typed upsert with an explicit `sequence`.
fn todo(key: &str, content: &str, sequence: i32) -> NewScratchpadNote {
    let mut n = NewScratchpadNote::new(key, content);
    n.note_type = "todo".to_string();
    n.sequence = Some(sequence);
    n
}

#[tokio::test]
async fn write_upserts_and_list_returns_all() {
    with_fixture("write_upserts_and_list_returns_all", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");

            let saved = pad
                .write("c1", &[note("goal", "ship it"), note("q", "which db")])
                .await
                .expect("batch write");
            assert_eq!(saved.len(), 2);

            let listed = pad.list("c1", None, 50).await.expect("list");
            assert_eq!(listed.len(), 2);

            // Re-writing an existing key updates content, not row count.
            pad.write("c1", &[note("goal", "ship it well")])
                .await
                .expect("upsert");
            let after = pad.list("c1", None, 50).await.expect("list after upsert");
            assert_eq!(after.len(), 2, "upsert must not create a duplicate row");
            let goal = after.iter().find(|n| n.key == "goal").expect("goal note");
            assert_eq!(goal.content, "ship it well");
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn get_many_fetches_requested_keys() {
    with_fixture("get_many_fetches_requested_keys", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            pad.write(
                "c1",
                &[note("a", "alpha"), note("b", "bravo"), note("c", "charlie")],
            )
            .await
            .expect("write");

            let got = pad
                .get_many("c1", &["a".to_string(), "c".to_string()], 50)
                .await
                .expect("get_many");
            let mut keys: Vec<String> = got.into_iter().map(|n| n.key).collect();
            keys.sort();
            assert_eq!(keys, vec!["a".to_string(), "c".to_string()]);
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn search_matches_full_text() {
    with_fixture("search_matches_full_text", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            pad.write(
                "c1",
                &[
                    note("deploy", "We will deploy the release on Friday"),
                    note("fruit", "unrelated apples and oranges"),
                ],
            )
            .await
            .expect("write");

            let hits = pad
                .search("c1", "deploy", Vec::new(), "", None, 50)
                .await
                .expect("search");
            assert_eq!(hits.len(), 1, "only the deploy note should match");
            assert_eq!(hits[0].key, "deploy");

            let none = pad
                .search("c1", "bicycle", Vec::new(), "", None, 50)
                .await
                .expect("search empty");
            assert!(none.is_empty());
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn delete_many_and_clear_return_counts() {
    with_fixture("delete_many_and_clear_return_counts", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            pad.write("c1", &[note("a", "x"), note("b", "y"), note("c", "z")])
                .await
                .expect("write");

            let deleted = pad
                .delete_many("c1", &["a".to_string(), "missing".to_string()])
                .await
                .expect("delete_many");
            assert_eq!(deleted, 1, "only the existing key is deleted");
            assert_eq!(pad.list("c1", None, 50).await.unwrap().len(), 2);

            let cleared = pad.clear("c1").await.expect("clear");
            assert_eq!(cleared, 2);
            assert!(pad.list("c1", None, 50).await.unwrap().is_empty());
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn deleting_conversation_cascades_to_notes() {
    with_fixture("deleting_conversation_cascades_to_notes", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            pad.write("c1", &[note("goal", "ship it")])
                .await
                .expect("write");
            assert_eq!(pad.list("c1", None, 50).await.unwrap().len(), 1);

            // Deleting the parent conversation must cascade to its notes.
            convs
                .delete(&ConversationId::from("c1"))
                .await
                .expect("delete conversation");
            assert!(
                pad.list("c1", None, 50).await.unwrap().is_empty(),
                "notes must be cascade-deleted with their conversation"
            );
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn cross_user_isolation() {
    with_fixture("cross_user_isolation", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        // Alice owns the conversation and writes a note.
        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("alice conv");
            pad.write("c1", &[note("goal", "alice secret")])
                .await
                .expect("alice write");
        })
        .await;

        // Bob, scoping to his own identity, can see / search / delete none of it.
        with_user_id(UserId::new("bob"), async {
            assert!(pad.list("c1", None, 50).await.unwrap().is_empty());
            assert!(
                pad.get_many("c1", &["goal".to_string()], 50)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                pad.search("c1", "secret", Vec::new(), "", None, 50)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                pad.delete_many("c1", &["goal".to_string()]).await.unwrap(),
                0,
                "bob must not be able to delete alice's notes"
            );
            assert_eq!(pad.clear("c1").await.unwrap(), 0);
        })
        .await;

        // Alice still has her note intact.
        with_user_id(UserId::new("alice"), async {
            assert_eq!(pad.list("c1", None, 50).await.unwrap().len(), 1);
        })
        .await;
        fx
    })
    .await;
}

// --- #809: cross-tenant write guard ----------------------------------------
//
// `write`'s `ON CONFLICT (conversation_id, owner_todo, note_key) DO UPDATE`
// has no `user_id` component in its conflict target (migration 031), so
// without a `WHERE` guard on the update, a second tenant writing a colliding
// key against another user's conversation id silently overwrites that row in
// place. `cross_user_isolation` above proves reads/deletes already fail
// closed; these prove the write path does too.

#[tokio::test]
async fn write_against_another_users_conversation_changes_nothing() {
    with_fixture(
        "write_against_another_users_conversation_changes_nothing",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());

            // Alice owns "c1" and writes her real note under key "goal".
            with_user_id(UserId::new("alice"), async {
                convs
                    .create(make_conversation("c1"))
                    .await
                    .expect("alice conv");
                pad.write("c1", &[note("goal", "alice's real plan")])
                    .await
                    .expect("alice write");
            })
            .await;

            // Bob writes against Alice's conversation id with a colliding note
            // key. The FK to conversations(id) is satisfied (it isn't scoped by
            // user), so the only thing standing between Bob and Alice's row is
            // the upsert's own guard. The call must not error, but must also
            // not report Alice's row as written/updated — it fails closed.
            let bob_result = with_user_id(UserId::new("bob"), async {
                pad.write("c1", &[note("goal", "bob's injected content")])
                    .await
                    .expect("bob's write must not error")
            })
            .await;
            assert!(
                bob_result.is_empty(),
                "a cross-tenant conflict must not report a written/updated row, got {bob_result:?}"
            );

            // Alice's content is unchanged, and no second row was created.
            with_user_id(UserId::new("alice"), async {
                let after = pad.list("c1", None, 50).await.expect("list");
                assert_eq!(
                    after.len(),
                    1,
                    "bob's write must not create a second row alongside alice's"
                );
                assert_eq!(
                    after[0].content, "alice's real plan",
                    "bob must not be able to overwrite alice's note content"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn write_to_own_conversation_still_upserts() {
    with_fixture("write_to_own_conversation_still_upserts", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            pad.write("c1", &[note("goal", "first draft")])
                .await
                .expect("initial write");

            // Re-writing the same key as the SAME user must still upsert in
            // place: the new user_id guard must not turn a normal same-tenant
            // upsert into a silent no-op.
            let updated = pad
                .write("c1", &[note("goal", "revised plan")])
                .await
                .expect("own-tenant upsert");
            assert_eq!(
                updated.len(),
                1,
                "own-tenant upsert must report the changed row"
            );
            assert_eq!(updated[0].content, "revised plan");

            let after = pad.list("c1", None, 50).await.expect("list");
            assert_eq!(after.len(), 1, "upsert must not create a duplicate row");
            assert_eq!(after[0].content, "revised plan");
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn single_tenant_write_is_unaffected() {
    with_fixture("single_tenant_write_is_unaffected", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        // No `with_user_id` scope at all: `current_user_id()` falls through to
        // the "default" sentinel for every call here, matching a single-tenant
        // desktop install with no JWT auth installed (design record
        // constraint: this path must not change behaviour).
        convs
            .create(make_conversation("c1"))
            .await
            .expect("create conv");
        let saved = pad
            .write("c1", &[note("goal", "ship it")])
            .await
            .expect("write");
        assert_eq!(saved.len(), 1);

        // Re-writing under the same (absent) scope still upserts in place: the
        // fail-closed guard compares the sentinel to itself and must still
        // match.
        let updated = pad
            .write("c1", &[note("goal", "ship it well")])
            .await
            .expect("upsert");
        assert_eq!(
            updated.len(),
            1,
            "single-tenant upsert must still report the changed row"
        );
        assert_eq!(updated[0].content, "ship it well");

        let after = pad.list("c1", None, 50).await.expect("list");
        assert_eq!(after.len(), 1, "upsert must not create a duplicate row");
        assert_eq!(after[0].content, "ship it well");
        fx
    })
    .await;
}

#[tokio::test]
async fn list_orders_by_type_then_sequence_nulls_last() {
    with_fixture(
        "list_orders_by_type_then_sequence_nulls_last",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                convs
                    .create(make_conversation("c1"))
                    .await
                    .expect("create conv");
                // Write todos out of sequence order, plus an unsequenced todo and a
                // plain note. Expect: type ascending ("note" < "todo"); within a
                // type, sequence ascending with NULLs last.
                let mut unseq = NewScratchpadNote::new("z", "no sequence");
                unseq.note_type = "todo".to_string();
                pad.write(
                    "c1",
                    &[
                        todo("c", "third", 3),
                        todo("a", "first", 1),
                        todo("b", "second", 2),
                        unseq,
                        note("n", "a plain note"),
                    ],
                )
                .await
                .expect("write");

                let listed = pad.list("c1", None, 50).await.expect("list");
                let keys: Vec<String> = listed.iter().map(|n| n.key.clone()).collect();
                assert_eq!(
                    keys,
                    vec!["n", "a", "b", "c", "z"],
                    "type then seq, nulls last"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn list_and_search_filter_by_type() {
    with_fixture("list_and_search_filter_by_type", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            pad.write(
                "c1",
                &[
                    todo("t1", "deploy the release", 1),
                    note("n1", "deploy notes from the meeting"),
                ],
            )
            .await
            .expect("write");

            // Type-filtered list returns only todos.
            let todos = pad.list("c1", Some("todo"), 50).await.expect("list todos");
            assert_eq!(todos.len(), 1);
            assert_eq!(todos[0].key, "t1");

            // Both notes match the FTS query; the type filter narrows to one.
            let all_hits = pad
                .search("c1", "deploy", Vec::new(), "", None, 50)
                .await
                .expect("search");
            assert_eq!(all_hits.len(), 2);
            let todo_hits = pad
                .search("c1", "deploy", Vec::new(), "", Some("todo"), 50)
                .await
                .expect("search todos");
            assert_eq!(todo_hits.len(), 1);
            assert_eq!(todo_hits[0].key, "t1");
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn rewrite_toggles_done_and_updates_fields() {
    with_fixture("rewrite_toggles_done_and_updates_fields", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            convs
                .create(make_conversation("c1"))
                .await
                .expect("create conv");
            let saved = pad.write("c1", &[todo("t1", "wire it", 1)]).await.unwrap();
            assert_eq!(saved[0].note_type, "todo");
            assert_eq!(saved[0].sequence, Some(1));
            assert!(!saved[0].done);

            // Re-writing the same key flips `done` (the check-off path) without
            // creating a duplicate row.
            let mut checked = todo("t1", "wire it", 1);
            checked.done = true;
            pad.write("c1", &[checked]).await.unwrap();

            let after = pad.list("c1", None, 50).await.unwrap();
            assert_eq!(after.len(), 1, "upsert keeps one row");
            assert!(after[0].done, "done flips on re-write");
        })
        .await;
        fx
    })
    .await;
}

// --- #287: owner_todo namespacing + ancestor-restricted snapshot reads ------

/// A subagent scope over the "c1" session pad: the given `owner_todo`
/// namespace, a snapshot `marker`, and the ancestor-namespace chain.
fn sub_scope(owner: &str, marker: &str, ancestors: &[&str]) -> SubagentScope {
    SubagentScope {
        session_conversation_id: ConversationId::from("c1"),
        owner_todo: owner.to_string(),
        visible_before: marker.to_string(),
        ancestors: ancestors.iter().map(|s| s.to_string()).collect(),
    }
}

fn key_set(notes: &[desktop_assistant_core::domain::ScratchpadNote]) -> HashSet<String> {
    notes.iter().map(|n| n.key.clone()).collect()
}

#[tokio::test]
async fn write_and_read_carry_owner_todo() {
    with_fixture("write_and_read_carry_owner_todo", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs.create(make_conversation("c1")).await.expect("conv");
            with_subagent_scope(sub_scope("1.1", "", &[]), async {
                pad.write("c1", &[note("finding", "x")])
                    .await
                    .expect("write");
            })
            .await;
            let all = pad.list("c1", None, 50).await.expect("list");
            let n = all
                .iter()
                .find(|n| n.key == "finding")
                .expect("finding note");
            assert_eq!(n.owner_todo, "1.1", "note carries its writer's owner_todo");
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn write_confined_to_owner_namespace() {
    with_fixture("write_confined_to_owner_namespace", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs.create(make_conversation("c1")).await.expect("conv");
            // Same note_key under root and under a subagent namespace = two rows
            // (an upsert collides only within one namespace).
            pad.write("c1", &[note("k", "root-val")])
                .await
                .expect("root write");
            with_subagent_scope(sub_scope("1.1", "", &[]), async {
                pad.write("c1", &[note("k", "sub-val")])
                    .await
                    .expect("sub write");
            })
            .await;
            let all = pad.list("c1", None, 50).await.expect("list");
            let mut owners: Vec<String> = all
                .iter()
                .filter(|n| n.key == "k")
                .map(|n| n.owner_todo.clone())
                .collect();
            owners.sort();
            assert_eq!(owners, vec!["".to_string(), "1.1".to_string()]);
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn snapshot_includes_own_and_descendant_excludes_sibling() {
    // #287 finding 1: the `id < marker` branch must be ancestor-restricted, not
    // namespace-blind. The sibling here is written BEFORE the marker (its id <
    // marker), so a naive namespace-blind predicate would wrongly include it.
    with_fixture(
        "snapshot_includes_own_and_descendant_excludes_sibling",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                // Parent (root) context, then a concurrent sibling 1.2 — both
                // BEFORE the child's spawn marker (ids < marker).
                pad.write("c1", &[note("ctx", "from-parent")])
                    .await
                    .expect("ctx");
                with_subagent_scope(sub_scope("1.2", "", &[]), async {
                    pad.write("c1", &[note("sib", "sibling")])
                        .await
                        .expect("sib");
                })
                .await;
                let marker = Uuid::now_v7().to_string();
                // Own + descendant writes AFTER the marker (ids > marker).
                with_subagent_scope(sub_scope("1.1", "", &[]), async {
                    pad.write("c1", &[note("own", "mine")]).await.expect("own");
                })
                .await;
                with_subagent_scope(sub_scope("1.1.1", "", &[]), async {
                    pad.write("c1", &[note("desc", "grandchild")])
                        .await
                        .expect("desc");
                })
                .await;
                // Read as subagent 1.1 with the snapshot marker; ancestors = root.
                let seen = with_subagent_scope(sub_scope("1.1", &marker, &[""]), async {
                    pad.list("c1", None, 50).await.expect("scoped list")
                })
                .await;
                let keys = key_set(&seen);
                assert!(keys.contains("ctx"), "ancestor pre-marker context visible");
                assert!(keys.contains("own"), "own namespace visible at any id");
                assert!(keys.contains("desc"), "descendant namespace visible");
                assert!(
                    !keys.contains("sib"),
                    "concurrent sibling 1.2 must NOT be visible even though its id < marker"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn subtree_prefix_does_not_leak_11_vs_111() {
    with_fixture("subtree_prefix_does_not_leak_11_vs_111", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs.create(make_conversation("c1")).await.expect("conv");
            // "1.11" is a SIBLING of "1.1", not a descendant; "1.1.9" is a real
            // descendant. The dot-delimited LIKE '1.1.%' must match only the latter.
            with_subagent_scope(sub_scope("1.11", "", &[]), async {
                pad.write("c1", &[note("eleven", "x")])
                    .await
                    .expect("eleven");
            })
            .await;
            with_subagent_scope(sub_scope("1.1.9", "", &[]), async {
                pad.write("c1", &[note("real_desc", "y")])
                    .await
                    .expect("desc");
            })
            .await;
            let marker = Uuid::now_v7().to_string();
            let seen = with_subagent_scope(sub_scope("1.1", &marker, &[""]), async {
                pad.list("c1", None, 50).await.expect("scoped list")
            })
            .await;
            let keys = key_set(&seen);
            assert!(
                keys.contains("real_desc"),
                "1.1.9 is a real descendant of 1.1"
            );
            assert!(
                !keys.contains("eleven"),
                "1.11 must NOT be seen as a descendant of 1.1 (dot boundary)"
            );
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn clear_only_wipes_own_namespace() {
    // Highest-severity guard: a subagent's `clear`/`delete all:true` must never
    // wipe the parent's pad.
    with_fixture("clear_only_wipes_own_namespace", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs.create(make_conversation("c1")).await.expect("conv");
            pad.write("c1", &[note("rootnote", "r")])
                .await
                .expect("root");
            with_subagent_scope(sub_scope("1.1", "", &[]), async {
                pad.write("c1", &[note("subnote", "s")]).await.expect("sub");
            })
            .await;
            let cleared = with_subagent_scope(sub_scope("1.1", "", &[]), async {
                pad.clear("c1").await.expect("clear")
            })
            .await;
            assert_eq!(cleared, 1, "subagent clear wipes only its own namespace");
            let keys = key_set(&pad.list("c1", None, 50).await.expect("list"));
            assert!(
                keys.contains("rootnote"),
                "parent pad survives a subagent clear"
            );
            assert!(!keys.contains("subnote"), "subagent's own note is cleared");
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn top_level_read_is_unbounded() {
    with_fixture("top_level_read_is_unbounded", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs.create(make_conversation("c1")).await.expect("conv");
            pad.write("c1", &[note("root", "r")]).await.expect("root");
            with_subagent_scope(sub_scope("1.1", "", &[]), async {
                pad.write("c1", &[note("sub", "s")]).await.expect("sub");
            })
            .await;
            // No scope installed => unbounded: the top-level sees every namespace.
            let keys = key_set(&pad.list("c1", None, 50).await.expect("list"));
            assert!(keys.contains("root") && keys.contains("sub"));
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn cross_user_isolation_still_holds_with_owner_todo() {
    with_fixture(
        "cross_user_isolation_still_holds_with_owner_todo",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                with_subagent_scope(sub_scope("1.1", "", &[]), async {
                    pad.write("c1", &[note("secret", "alice-only")])
                        .await
                        .expect("write");
                })
                .await;
            })
            .await;
            // Bob, even reading the same conversation id under a matching scope, sees
            // and deletes nothing — the user_id=$1 guard holds regardless of owner_todo.
            with_user_id(UserId::new("bob"), async {
                let marker = Uuid::now_v7().to_string();
                let seen = with_subagent_scope(sub_scope("1.1", &marker, &[""]), async {
                    pad.list("c1", None, 50).await.expect("list")
                })
                .await;
                assert!(seen.is_empty(), "bob sees none of alice's rows");
                let cleared = with_subagent_scope(sub_scope("1.1", "", &[]), async {
                    pad.clear("c1").await.expect("clear")
                })
                .await;
                assert_eq!(cleared, 0, "bob deletes none of alice's rows");
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn migration_031_is_idempotent() {
    with_fixture("migration_031_is_idempotent", |fx| async move {
        // The runner re-executes every migration on every startup with no
        // version table, so a second run on the same schema must succeed.
        run_migrations(&fx.pool)
            .await
            .expect("second run_migrations is idempotent");
        fx
    })
    .await;
}

// --- #287 slice 4: delete_owner_subtree (cascade primitive) -----------------

/// Write one note with the given key under `owner` on conversation `conv`.
async fn write_owned(pad: &PgScratchpadStore, conv: &str, owner: &str, key: &str) {
    with_subagent_scope(sub_scope(owner, "", &[]), async {
        pad.write(conv, &[note(key, owner)])
            .await
            .expect("write_owned");
    })
    .await;
}

fn owner_set(notes: &[desktop_assistant_core::domain::ScratchpadNote]) -> HashSet<String> {
    notes.iter().map(|n| n.owner_todo.clone()).collect()
}

#[tokio::test]
async fn delete_owner_subtree_removes_self_and_descendants() {
    with_fixture(
        "delete_owner_subtree_removes_self_and_descendants",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                for owner in ["", "1", "1.1", "1.2.3", "2"] {
                    write_owned(&pad, "c1", owner, "k").await;
                }
                let n = pad.delete_owner_subtree("c1", "1").await.expect("delete");
                assert_eq!(n, 3, "subtree of '1' = 1, 1.1, 1.2.3");
                assert_eq!(
                    owner_set(&pad.list("c1", None, 50).await.expect("list")),
                    HashSet::from(["".to_string(), "2".to_string()])
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn delete_owner_subtree_is_dot_boundary_safe() {
    with_fixture(
        "delete_owner_subtree_is_dot_boundary_safe",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                for owner in ["1", "1.1", "10", "11", "2"] {
                    write_owned(&pad, "c1", owner, "k").await;
                }
                let n = pad.delete_owner_subtree("c1", "1").await.expect("delete");
                assert_eq!(n, 2, "'1' matches only '1' and '1.1', never '10'/'11'/'2'");
                assert_eq!(
                    owner_set(&pad.list("c1", None, 50).await.expect("list")),
                    HashSet::from(["10".to_string(), "11".to_string(), "2".to_string()])
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn delete_owner_subtree_is_user_and_conversation_scoped_fail_closed() {
    with_fixture(
        "delete_owner_subtree_is_user_and_conversation_scoped_fail_closed",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("c1");
                convs.create(make_conversation("c2")).await.expect("c2");
                write_owned(&pad, "c1", "1", "k").await;
                write_owned(&pad, "c2", "1", "k").await;
            })
            .await;
            // Bob owns none of alice's rows: his cascade deletes nothing.
            with_user_id(UserId::new("bob"), async {
                let n = pad
                    .delete_owner_subtree("c1", "1")
                    .await
                    .expect("bob delete");
                assert_eq!(n, 0, "cross-user cascade is fail-closed");
            })
            .await;
            // Alice's cascade on c1 leaves c2 untouched (conversation-scoped).
            with_user_id(UserId::new("alice"), async {
                let n = pad
                    .delete_owner_subtree("c1", "1")
                    .await
                    .expect("alice delete");
                assert_eq!(n, 1);
                assert_eq!(pad.list("c2", None, 50).await.expect("c2 list").len(), 1);
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn delete_owner_subtree_idempotent() {
    with_fixture("delete_owner_subtree_idempotent", |fx| async move {
        let convs = PgConversationStore::new(fx.pool.clone());
        let pad = PgScratchpadStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            convs.create(make_conversation("c1")).await.expect("conv");
            write_owned(&pad, "c1", "1", "k").await;
            write_owned(&pad, "c1", "1.1", "k").await;
            assert_eq!(pad.delete_owner_subtree("c1", "1").await.expect("del1"), 2);
            assert_eq!(
                pad.delete_owner_subtree("c1", "1").await.expect("del2"),
                0,
                "second cascade is a no-op"
            );
        })
        .await;
        fx
    })
    .await;
}

// --- #1104: a note attaches a knowledge entry ------------------------------

/// Insert a knowledge entry owned by the current user, so a note may attach it.
/// Uses the knowledge store rather than raw SQL, so the row is exactly what the
/// product writes.
async fn write_entry(pool: &PgPool, id: &str, content: &str) {
    use desktop_assistant_core::domain::KnowledgeEntry;
    use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
    desktop_assistant_storage::PgKnowledgeBaseStore::new(pool.clone())
        .write(KnowledgeEntry::new(id, content, vec![]))
        .await
        .expect("write knowledge entry");
}

/// A note that attaches `entry_id`.
fn note_attaching(key: &str, content: &str, entry_id: &str) -> NewScratchpadNote {
    let mut n = NewScratchpadNote::new(key, content);
    n.knowledge_entry_id = Some(entry_id.to_string());
    n
}

#[tokio::test]
async fn pinned_reference_stays_within_the_calling_user() {
    // Row-level security is a backstop the table owner bypasses, so the
    // `user_id` predicate is the real guard. Bob must neither read alice's
    // attachment nor repair it out from under her.
    with_fixture(
        "pinned_reference_stays_within_the_calling_user",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());

            let alice_note_id = with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-alice", "alice's durable fact").await;
                let saved = pad
                    .write(
                        "c1",
                        &[note_attaching("deploy-target", "settled", "kb-alice")],
                    )
                    .await
                    .expect("alice write");
                pad.set_pinned("c1", &["deploy-target".to_string()], true)
                    .await
                    .expect("alice pin");
                saved[0].id.clone()
            })
            .await;

            with_user_id(UserId::new("bob"), async {
                assert!(
                    pad.list("c1", None, 50).await.expect("bob list").is_empty(),
                    "bob must not see alice's note, attachment and all"
                );
                assert_eq!(
                    pad.release_knowledge_references("c1", std::slice::from_ref(&alice_note_id))
                        .await
                        .expect("bob release"),
                    0,
                    "bob must not repair alice's row"
                );
            })
            .await;

            with_user_id(UserId::new("alice"), async {
                let notes = pad.list("c1", None, 50).await.expect("alice list");
                assert_eq!(
                    notes[0].knowledge_entry_id.as_deref(),
                    Some("kb-alice"),
                    "alice's attachment survives bob's attempt"
                );
                assert!(notes[0].pinned, "and so does her pin");
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn pinned_reference_stays_within_the_subagent_namespace() {
    // Confinement matches what `set_pinned` already enforces: a subagent may
    // attach and pin in its own namespace, never in the parent's.
    with_fixture(
        "pinned_reference_stays_within_the_subagent_namespace",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-1", "the durable fact").await;

                // The parent attaches and pins in the root namespace.
                pad.write("c1", &[note_attaching("shared-key", "parent", "kb-1")])
                    .await
                    .expect("parent write");
                pad.set_pinned("c1", &["shared-key".to_string()], true)
                    .await
                    .expect("parent pin");

                // A subagent writing the SAME key gets its own row in its own
                // namespace, and its pin does not reach the parent's.
                with_subagent_scope(sub_scope("1.1", "", &[]), async {
                    pad.write("c1", &[note_attaching("shared-key", "child", "kb-1")])
                        .await
                        .expect("child write");
                    pad.set_pinned("c1", &["shared-key".to_string()], true)
                        .await
                        .expect("child pin");
                })
                .await;

                let notes = pad.list("c1", None, 50).await.expect("list");
                let parent = notes
                    .iter()
                    .find(|n| n.owner_todo.is_empty())
                    .expect("the parent's row");
                let child = notes
                    .iter()
                    .find(|n| n.owner_todo == "1.1")
                    .expect("the child's row");
                assert_eq!(parent.content, "parent", "the child must not overwrite it");
                assert_eq!(child.content, "child");
                assert_eq!(parent.knowledge_entry_id.as_deref(), Some("kb-1"));
                assert_eq!(child.knowledge_entry_id.as_deref(), Some("kb-1"));

                // The child's own unpin reaches only its own row.
                with_subagent_scope(sub_scope("1.1", "", &[]), async {
                    pad.set_pinned("c1", &["shared-key".to_string()], false)
                        .await
                        .expect("child unpin");
                })
                .await;
                let notes = pad.list("c1", None, 50).await.expect("list again");
                assert!(
                    notes
                        .iter()
                        .find(|n| n.owner_todo.is_empty())
                        .expect("parent row")
                        .pinned,
                    "a subagent must not unpin the parent's note"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn a_goal_note_and_a_todo_note_are_both_pinnable() {
    // `set_pinned` filters on `note_key` only. Locking that in stops a later
    // change quietly restricting pinning to one kind of note.
    with_fixture(
        "a_goal_note_and_a_todo_note_are_both_pinnable",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                pad.write("c1", &[note("goal", "ship it"), todo("1", "wire it", 1)])
                    .await
                    .expect("write");
                assert_eq!(
                    pad.set_pinned("c1", &["goal".to_string(), "1".to_string()], true)
                        .await
                        .expect("pin both"),
                    2,
                    "both a goal note and a todo note must be pinnable"
                );
                let notes = pad.list("c1", None, 50).await.expect("list");
                assert!(notes.iter().all(|n| n.pinned), "both rows are pinned");
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn a_note_keeps_its_attachment_when_its_content_is_rewritten() {
    // `None` preserves. A caller that rewrites the text and knows nothing about
    // the attachment must not silently drop it.
    with_fixture(
        "a_note_keeps_its_attachment_when_its_content_is_rewritten",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-1", "the durable fact").await;
                pad.write("c1", &[note_attaching("deploy-target", "first", "kb-1")])
                    .await
                    .expect("attach");
                let saved = pad
                    .write("c1", &[note("deploy-target", "rewritten")])
                    .await
                    .expect("rewrite");
                assert_eq!(saved[0].content, "rewritten");
                assert_eq!(
                    saved[0].knowledge_entry_id.as_deref(),
                    Some("kb-1"),
                    "an ordinary rewrite must not drop the attachment"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn releasing_a_reference_clears_the_attachment_and_the_pin() {
    // The repair the render path calls when an entry no longer resolves, plus
    // its idempotence: a second call changes nothing.
    with_fixture(
        "releasing_a_reference_clears_the_attachment_and_the_pin",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-1", "the durable fact").await;
                let saved = pad
                    .write("c1", &[note_attaching("deploy-target", "settled", "kb-1")])
                    .await
                    .expect("attach");
                pad.set_pinned("c1", &["deploy-target".to_string()], true)
                    .await
                    .expect("pin");
                let id = saved[0].id.clone();

                assert_eq!(
                    pad.release_knowledge_references("c1", std::slice::from_ref(&id))
                        .await
                        .expect("release"),
                    1
                );
                let notes = pad.list("c1", None, 50).await.expect("list");
                assert_eq!(notes[0].knowledge_entry_id, None, "attachment cleared");
                assert!(!notes[0].pinned, "and the pin released with it");
                assert_eq!(notes[0].content, "settled", "the model's own words survive");

                assert_eq!(
                    pad.release_knowledge_references("c1", &[id])
                        .await
                        .expect("release again"),
                    0,
                    "the repair is idempotent"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn deleting_a_knowledge_entry_leaves_the_attachment_for_the_render_to_repair() {
    // There is deliberately no foreign key. `ON DELETE SET NULL` would clear the
    // column the instant the entry row went, and the note would keep its pin,
    // render nothing under it, and never be told - because the evidence the
    // render needs to notice would be gone. The id must survive the delete so a
    // hard delete and a soft delete take the same repair path.
    with_fixture(
        "deleting_a_knowledge_entry_leaves_the_attachment_for_the_render_to_repair",
        |fx| async move {
            use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            let kb = desktop_assistant_storage::PgKnowledgeBaseStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-1", "the durable fact").await;
                pad.write("c1", &[note_attaching("deploy-target", "settled", "kb-1")])
                    .await
                    .expect("attach");
                pad.set_pinned("c1", &["deploy-target".to_string()], true)
                    .await
                    .expect("pin");

                kb.delete("kb-1").await.expect("delete entry");

                let notes = pad.list("c1", None, 50).await.expect("list");
                assert_eq!(
                    notes[0].knowledge_entry_id.as_deref(),
                    Some("kb-1"),
                    "the id must survive, or the render cannot see the pin is empty"
                );
                assert!(
                    notes[0].pinned,
                    "and the pin is still the render's to release"
                );
                // The entry itself is gone, so the render's read finds nothing
                // and repairs the note.
                assert!(
                    kb.get("kb-1").await.expect("get").is_none(),
                    "the entry really is gone"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn releasing_a_reference_stays_within_the_subagent_namespace() {
    // A subagent's read spans its ancestors, so without confinement a subagent
    // round would clear the parent's pin and consume the one line that says a
    // pin was released - in the subagent's block, where the parent never sees
    // it. Confined, the parent repairs its own on its own next round.
    with_fixture(
        "releasing_a_reference_stays_within_the_subagent_namespace",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-1", "the durable fact").await;
                let saved = pad
                    .write("c1", &[note_attaching("deploy-target", "settled", "kb-1")])
                    .await
                    .expect("parent attach");
                pad.set_pinned("c1", &["deploy-target".to_string()], true)
                    .await
                    .expect("parent pin");
                let parent_note_id = saved[0].id.clone();

                let released = with_subagent_scope(sub_scope("1.1", "", &[]), async {
                    pad.release_knowledge_references("c1", std::slice::from_ref(&parent_note_id))
                        .await
                        .expect("child release")
                })
                .await;
                assert_eq!(released, 0, "a subagent must not repair an ancestor's row");

                let notes = pad.list("c1", None, 50).await.expect("list");
                assert_eq!(notes[0].knowledge_entry_id.as_deref(), Some("kb-1"));
                assert!(notes[0].pinned, "the parent's pin is untouched");

                assert_eq!(
                    pad.release_knowledge_references("c1", std::slice::from_ref(&parent_note_id))
                        .await
                        .expect("parent release"),
                    1,
                    "the owner repairs its own"
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn a_top_level_round_repairs_a_subagent_notes_dangling_attachment() {
    // The top-level read is namespace-blind, so a top-level round SEES a
    // subagent's pinned note. If it could not also repair it, that note would
    // keep a dead attachment for the life of the conversation, hold a slot of
    // the pin cap, and cost a knowledge read every round - and the model could
    // not clear it either, because the pin and delete verbs are confined to the
    // caller's own namespace. So the repair reaches the caller's own subtree,
    // and the root's subtree is the whole pad.
    with_fixture(
        "a_top_level_round_repairs_a_subagent_notes_dangling_attachment",
        |fx| async move {
            let convs = PgConversationStore::new(fx.pool.clone());
            let pad = PgScratchpadStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                convs.create(make_conversation("c1")).await.expect("conv");
                write_entry(&fx.pool, "kb-1", "the durable fact").await;

                let child_note_id = with_subagent_scope(sub_scope("1", "", &[]), async {
                    let saved = pad
                        .write("c1", &[note_attaching("api-contract", "child", "kb-1")])
                        .await
                        .expect("child attach");
                    pad.set_pinned("c1", &["api-contract".to_string()], true)
                        .await
                        .expect("child pin");
                    saved[0].id.clone()
                })
                .await;

                // The top level sees it (namespace-blind read) and repairs it.
                assert!(
                    pad.list("c1", None, 50)
                        .await
                        .expect("top-level list")
                        .iter()
                        .any(|n| n.owner_todo == "1" && n.pinned),
                    "precondition: the top level sees the subagent's pinned note"
                );
                assert_eq!(
                    pad.release_knowledge_references("c1", std::slice::from_ref(&child_note_id))
                        .await
                        .expect("top-level release"),
                    1,
                    "the root subtree is the whole pad, so nothing is left stuck"
                );
                let notes = pad.list("c1", None, 50).await.expect("list");
                assert_eq!(notes[0].knowledge_entry_id, None);
                assert!(!notes[0].pinned, "and the slot of the pin cap is freed");
            })
            .await;
            fx
        },
    )
    .await;
}
