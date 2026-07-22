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

            let hits = pad.search("c1", "deploy", None, 50).await.expect("search");
            assert_eq!(hits.len(), 1, "only the deploy note should match");
            assert_eq!(hits[0].key, "deploy");

            let none = pad
                .search("c1", "bicycle", None, 50)
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
                pad.search("c1", "secret", None, 50)
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
            let all_hits = pad.search("c1", "deploy", None, 50).await.expect("search");
            assert_eq!(all_hits.len(), 2);
            let todo_hits = pad
                .search("c1", "deploy", Some("todo"), 50)
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
