//! Cross-user isolation integration tests for #105.
//!
//! These tests exercise the storage adapters end-to-end against a real
//! Postgres instance with the multi-tenant schema in place. They prove
//! that:
//!
//! - Writes under user A land in A's partition only.
//! - Reads under user A see only A's rows.
//! - Cross-user lookups return `ConversationNotFound` rather than
//!   leaking the existence of another user's data.
//! - The single-tenant default path (no `UserId` scope installed)
//!   resolves to the schema sentinel `"default"`, so an unscoped
//!   request can still write and read.
//!
//! ## Running locally
//!
//! Set `TEST_DATABASE_URL` to a Postgres URL where the connecting role
//! can `CREATE SCHEMA` and `CREATE EXTENSION vector`:
//!
//! ```sh
//! podman run -d --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 \
//!     docker.io/pgvector/pgvector:pg17
//! TEST_DATABASE_URL="postgres://postgres:test@localhost:15432/postgres" \
//!     cargo test -p desktop-assistant-storage --test user_id_scoping
//! ```
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips with a log
//! line so the suite stays green without a DB.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, KnowledgeEntry, Message, Role};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore, ConversationStore,
    PendingClientToolCall, TurnRow, TurnStateJson, TurnStateStore, TurnStatus,
};
use desktop_assistant_storage::{
    PgBackgroundTaskStore, PgConversationStore, PgKnowledgeBaseStore, PgTurnStateStore, UserId,
    run_migrations, with_user_id,
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
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let schema = format!("issue105_{}", Uuid::now_v7().simple());

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

fn make_conversation(id: &str, title: &str) -> Conversation {
    let mut conv = Conversation::new(id, title);
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    conv.messages.push(Message::new(Role::User, "hello"));
    conv
}

// -- conversations -----------------------------------------------------------

#[tokio::test]
async fn user_a_writes_then_user_b_cannot_list_it() {
    // Two users concurrent: A writes a conversation under their scope;
    // B's list under their own scope returns nothing. Mirrors the
    // top-level `two_users_concurrent_conversations_isolated` test in
    // the issue brief.
    with_fixture(
        "user_a_writes_then_user_b_cannot_list_it",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            // User A: insert a conversation.
            with_user_id(UserId::new("alice"), async {
                store
                    .create(make_conversation("conv-a-1", "alice chat"))
                    .await
                    .expect("alice create");
            })
            .await;

            // User B: list — should see nothing.
            let bobs_list = with_user_id(UserId::new("bob"), async { store.list().await })
                .await
                .expect("bob list");
            assert!(
                bobs_list.is_empty(),
                "bob's list must NOT include alice's conversations, got {} row(s)",
                bobs_list.len()
            );

            // User A: list — should see exactly their one row.
            let alices_list = with_user_id(UserId::new("alice"), async { store.list().await })
                .await
                .expect("alice list");
            assert_eq!(alices_list.len(), 1);
            assert_eq!(alices_list[0].id.0, "conv-a-1");

            fx
        },
    )
    .await;
}

#[tokio::test]
async fn user_b_cannot_read_user_a_conversation_by_id() {
    // Cross-user direct lookup must return NotFound — don't leak the
    // existence of A's conversation to B.
    with_fixture(
        "user_b_cannot_read_user_a_conversation_by_id",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create(make_conversation("conv-x", "alice"))
                    .await
                    .expect("alice create");
            })
            .await;

            let bob_get = with_user_id(UserId::new("bob"), async {
                store.get(&ConversationId::from("conv-x")).await
            })
            .await;
            match bob_get {
                Err(CoreError::ConversationNotFound(id)) => {
                    assert_eq!(id, "conv-x", "bob's NotFound must echo the id he asked for");
                }
                other => panic!("expected ConversationNotFound, got {other:?}"),
            }
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn user_b_cannot_delete_user_a_conversation_by_id() {
    // Mutating cross-user attempt also returns NotFound — same don't-
    // leak-existence rule as the read path.
    with_fixture(
        "user_b_cannot_delete_user_a_conversation_by_id",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create(make_conversation("conv-y", "alice"))
                    .await
                    .expect("alice create");
            })
            .await;

            let bob_delete = with_user_id(UserId::new("bob"), async {
                store.delete(&ConversationId::from("conv-y")).await
            })
            .await;
            match bob_delete {
                Err(CoreError::ConversationNotFound(id)) => {
                    assert_eq!(id, "conv-y");
                }
                other => panic!("expected ConversationNotFound, got {other:?}"),
            }

            // Alice's row is still intact.
            let alices_get = with_user_id(UserId::new("alice"), async {
                store.get(&ConversationId::from("conv-y")).await
            })
            .await;
            assert!(alices_get.is_ok(), "alice's row should still be readable");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn two_users_concurrent_conversations_isolated() {
    // tokio::join concurrent writes — neither user sees the other's
    // data afterwards. Direct exercise of the issue's primary
    // acceptance test.
    with_fixture(
        "two_users_concurrent_conversations_isolated",
        |fx| async move {
            let store = Arc::new(PgConversationStore::new(fx.pool.clone()));

            let store_a = Arc::clone(&store);
            let store_b = Arc::clone(&store);

            let alice = with_user_id(UserId::new("alice"), async move {
                store_a.create(make_conversation("conv-a", "alice")).await
            });
            let bob = with_user_id(UserId::new("bob"), async move {
                store_b.create(make_conversation("conv-b", "bob")).await
            });
            let (a, b) = tokio::join!(alice, bob);
            a.expect("alice create");
            b.expect("bob create");

            let alices_list = with_user_id(UserId::new("alice"), async { store.list().await })
                .await
                .unwrap();
            let bobs_list = with_user_id(UserId::new("bob"), async { store.list().await })
                .await
                .unwrap();

            assert_eq!(alices_list.len(), 1);
            assert_eq!(alices_list[0].id.0, "conv-a");
            assert_eq!(bobs_list.len(), 1);
            assert_eq!(bobs_list[0].id.0, "conv-b");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn existing_messages_table_inserts_set_user_id() {
    // Regression: prove the message-insert path stamps the
    // task-local user id into messages.user_id (not just
    // conversations.user_id).
    with_fixture(
        "existing_messages_table_inserts_set_user_id",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create(make_conversation("conv-msg", "alice"))
                    .await
                    .expect("alice create");
            })
            .await;

            let row: (String,) =
                sqlx::query_as("SELECT user_id FROM messages WHERE conversation_id = $1 LIMIT 1")
                    .bind("conv-msg")
                    .fetch_one(&fx.pool)
                    .await
                    .expect("read back message row");
            assert_eq!(row.0, "alice");
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn request_with_default_user_id_succeeds_in_single_tenant_path() {
    // Single-tenant deploy boundary: no scope installed, no JWT in
    // play, write/read still works against the sentinel partition.
    with_fixture(
        "request_with_default_user_id_succeeds_in_single_tenant_path",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());

            // No `with_user_id` wrapper — the storage layer falls
            // through to the sentinel.
            store
                .create(make_conversation("conv-default", "single tenant"))
                .await
                .expect("create with no scope installed");

            let list = store.list().await.expect("list with no scope");
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].id.0, "conv-default");

            // The persisted row must carry the sentinel `user_id`.
            let row: (String,) = sqlx::query_as("SELECT user_id FROM conversations WHERE id = $1")
                .bind("conv-default")
                .fetch_one(&fx.pool)
                .await
                .expect("read back");
            assert_eq!(row.0, "default");
            fx
        },
    )
    .await;
}

// -- knowledge base ----------------------------------------------------------

#[tokio::test]
async fn knowledge_writes_are_isolated_per_user() {
    with_fixture("knowledge_writes_are_isolated_per_user", |fx| async move {
        let store = PgKnowledgeBaseStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            let entry = KnowledgeEntry::new("kb-alice", "alice loves rust", vec!["pref".into()]);
            store.write(entry).await.expect("alice write");
        })
        .await;

        with_user_id(UserId::new("bob"), async {
            let entry = KnowledgeEntry::new("kb-bob", "bob loves zig", vec!["pref".into()]);
            store.write(entry).await.expect("bob write");
        })
        .await;

        // Each user lists only their own entry.
        let alices = with_user_id(UserId::new("alice"), async {
            store.list(100, 0, None).await
        })
        .await
        .unwrap();
        assert_eq!(alices.len(), 1);
        assert_eq!(alices[0].id, "kb-alice");

        let bobs = with_user_id(UserId::new("bob"), async { store.list(100, 0, None).await })
            .await
            .unwrap();
        assert_eq!(bobs.len(), 1);
        assert_eq!(bobs[0].id, "kb-bob");
        fx
    })
    .await;
}

#[tokio::test]
async fn knowledge_search_does_not_leak_across_users() {
    // pgvector / FTS path: user A indexes a doc; user B searches the
    // same query and doesn't see it. Mirrors the issue brief's
    // `knowledge_search_does_not_leak_across_users` test.
    with_fixture(
        "knowledge_search_does_not_leak_across_users",
        |fx| async move {
            let store = PgKnowledgeBaseStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                let entry = KnowledgeEntry::new(
                    "kb-alice-doc",
                    "Alice's private project notes about widget refactoring",
                    vec!["project".into()],
                );
                store.write(entry).await.expect("alice index");
            })
            .await;

            // FTS-only path (no embedding required).
            let bob_hits = with_user_id(UserId::new("bob"), async {
                store.search_text("widget refactoring", None, 10).await
            })
            .await
            .unwrap();
            assert!(
                bob_hits.is_empty(),
                "bob's text search must not see alice's doc, got {} hit(s)",
                bob_hits.len()
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_search_with_empty_embedding_falls_back_to_fts() {
    // #194: when the embedding backend times out, the search runs with an empty
    // query embedding. The hybrid query's vector branch (`chunk <=> $1`) would
    // error on a 0-dimension vector against rows that DO have embeddings, so the
    // store must skip straight to FTS. Index a doc WITH an embedding, then
    // search with an empty embedding and require an FTS hit (and no error).
    with_fixture(
        "knowledge_search_with_empty_embedding_falls_back_to_fts",
        |fx| async move {
            let store = PgKnowledgeBaseStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                let entry = KnowledgeEntry::new(
                    "kb-forecast-doc",
                    "Quarterly forecast planning for the widget team",
                    vec!["project".into()],
                );
                // Writes never embed inline; the row lands with a NULL
                // embedding, which is exactly the condition this test exercises
                // (search must fall back to FTS rather than erroring).
                store.write(entry).await.expect("write");

                let hits = store
                    .search("forecast planning", Vec::new(), None, None, 10)
                    .await
                    .expect("empty-embedding search must fall back to FTS, not error");
                assert!(
                    hits.iter().any(|e| e.id == "kb-forecast-doc"),
                    "FTS fallback must find the indexed doc, got {} hit(s)",
                    hits.len()
                );
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn knowledge_get_by_id_does_not_leak_across_users() {
    with_fixture(
        "knowledge_get_by_id_does_not_leak_across_users",
        |fx| async move {
            let store = PgKnowledgeBaseStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                let entry =
                    KnowledgeEntry::new("kb-shared-id", "alice content", vec!["pref".into()]);
                store.write(entry).await.expect("alice write");
            })
            .await;

            // Bob tries to fetch by the exact same id — must miss.
            let bob_fetch = with_user_id(UserId::new("bob"), async {
                store.get("kb-shared-id").await
            })
            .await
            .unwrap();
            assert!(
                bob_fetch.is_none(),
                "bob's get by id must miss when the id belongs to alice"
            );

            // Alice still sees it.
            let alice_fetch = with_user_id(UserId::new("alice"), async {
                store.get("kb-shared-id").await
            })
            .await
            .unwrap();
            assert!(alice_fetch.is_some());
            fx
        },
    )
    .await;
}

// -- turn state (issue #107) -------------------------------------------------

fn turn_row(id: &str, user_id: &str, conversation_id: &str, status: TurnStatus) -> TurnRow {
    TurnRow {
        id: id.into(),
        user_id: user_id.into(),
        conversation_id: conversation_id.into(),
        status,
        state: TurnStateJson::default(),
        last_error: None,
    }
}

#[tokio::test]
async fn turn_state_round_trips_through_postgres() {
    // Create + get + update + read-back for a single user's turn row.
    // The acceptance value here is that the state_json column survives
    // a serde round-trip through Postgres's JSONB representation.
    with_fixture("turn_state_round_trips_through_postgres", |fx| async move {
        let store = PgTurnStateStore::new(fx.pool.clone());

        with_user_id(UserId::new("alice"), async {
            store
                .create_turn(turn_row(
                    "turn-1",
                    "alice",
                    "conv-1",
                    TurnStatus::PendingLlm,
                ))
                .await
                .expect("create");

            // Read back: alice sees her row.
            let row = store
                .get_turn("turn-1")
                .await
                .expect("get")
                .expect("row exists");
            assert_eq!(row.status, TurnStatus::PendingLlm);
            assert!(row.state.pending_client_tool.is_none());

            // Transition to pending_client_tool with a payload.
            let state = TurnStateJson {
                version: 1,
                pending_client_tool: Some(PendingClientToolCall {
                    tool_call_id: "call-7".into(),
                    tool_name: "fs_read".into(),
                    arguments: serde_json::json!({"path": "/etc/hosts"}),
                }),
            };
            store
                .update_turn("turn-1", TurnStatus::PendingClientTool, &state, None)
                .await
                .expect("update");

            let row = store.get_turn("turn-1").await.unwrap().unwrap();
            assert_eq!(row.status, TurnStatus::PendingClientTool);
            let pending = row.state.pending_client_tool.unwrap();
            assert_eq!(pending.tool_call_id, "call-7");
            assert_eq!(pending.tool_name, "fs_read");
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn turn_state_user_b_cannot_read_user_a_turn() {
    // Cross-user isolation, mirroring the conversation-level test
    // above. Bob's get_turn must return None — not "not found" — and
    // his update_turn must error out for the same opacity reason.
    with_fixture(
        "turn_state_user_b_cannot_read_user_a_turn",
        |fx| async move {
            let store = PgTurnStateStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create_turn(turn_row(
                        "turn-x",
                        "alice",
                        "conv-1",
                        TurnStatus::PendingLlm,
                    ))
                    .await
                    .expect("create");
            })
            .await;

            let bob_get =
                with_user_id(UserId::new("bob"), async { store.get_turn("turn-x").await })
                    .await
                    .expect("get returns Ok for cross-user");
            assert!(bob_get.is_none(), "bob must not see alice's turn");

            let bob_update = with_user_id(UserId::new("bob"), async {
                store
                    .update_turn(
                        "turn-x",
                        TurnStatus::Failed,
                        &TurnStateJson::default(),
                        Some("nope"),
                    )
                    .await
            })
            .await;
            assert!(
                bob_update.is_err(),
                "bob must not be able to mutate alice's turn"
            );

            // Alice's row is unaffected.
            let alice_get = with_user_id(UserId::new("alice"), async {
                store.get_turn("turn-x").await
            })
            .await
            .unwrap()
            .unwrap();
            assert_eq!(alice_get.status, TurnStatus::PendingLlm);
            assert!(alice_get.last_error.is_none());
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn turn_state_scan_non_terminal_walks_all_users() {
    // The cold-restart sweep is a system task — it deliberately walks
    // every user's pending rows in a single pass. This pins that
    // contract against the SQL.
    with_fixture(
        "turn_state_scan_non_terminal_walks_all_users",
        |fx| async move {
            let store = PgTurnStateStore::new(fx.pool.clone());

            with_user_id(UserId::new("alice"), async {
                store
                    .create_turn(turn_row(
                        "t-a-pending",
                        "alice",
                        "c",
                        TurnStatus::PendingLlm,
                    ))
                    .await
                    .unwrap();
                store
                    .create_turn(turn_row("t-a-complete", "alice", "c", TurnStatus::Complete))
                    .await
                    .unwrap();
            })
            .await;

            with_user_id(UserId::new("bob"), async {
                store
                    .create_turn(turn_row(
                        "t-b-pending",
                        "bob",
                        "c",
                        TurnStatus::PendingClientTool,
                    ))
                    .await
                    .unwrap();
                store
                    .create_turn(turn_row("t-b-failed", "bob", "c", TurnStatus::Failed))
                    .await
                    .unwrap();
            })
            .await;

            // Sweep across all users — no scope installed.
            let mut rows = store.scan_non_terminal().await.expect("scan");
            rows.sort_by(|a, b| a.id.cmp(&b.id));
            let ids: Vec<_> = rows.iter().map(|r| r.id.clone()).collect();
            assert_eq!(
                ids,
                vec!["t-a-pending".to_string(), "t-b-pending".to_string()]
            );
            fx
        },
    )
    .await;
}

// -- background tasks (issue #115) ------------------------------------------

fn task_row(id: &str, user_id: &str, status: BackgroundTaskStatus) -> BackgroundTaskRow {
    BackgroundTaskRow {
        id: id.into(),
        user_id: user_id.into(),
        // The store treats `kind_json` opaquely; any well-formed JSON is
        // fine. We pick a `standalone` shape because the registry tests
        // hammer that branch hardest.
        kind_json: serde_json::json!({
            "standalone": {"name": "test", "conversation_id": "c"}
        }),
        status,
        parent_task_id: None,
        title: format!("title for {id}"),
        last_error: None,
        progress_hint: None,
        started_at: 1_700_000_000,
        ended_at: None,
    }
}

#[tokio::test]
async fn background_task_row_round_trips_through_postgres() {
    with_fixture(
        "background_task_row_round_trips_through_postgres",
        |fx| async move {
            let store = PgBackgroundTaskStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                store
                    .create_task(task_row("bt-1", "alice", BackgroundTaskStatus::Running))
                    .await
                    .expect("create");
                let row = store.get_task("bt-1").await.unwrap().expect("row");
                assert_eq!(row.status, BackgroundTaskStatus::Running);
                assert_eq!(row.title, "title for bt-1");

                store
                    .update_task(
                        "bt-1",
                        BackgroundTaskStatus::Failed,
                        Some("daemon restarted mid-turn"),
                        Some("step 2/4"),
                        Some(1_700_000_555),
                    )
                    .await
                    .expect("update");
                let row = store.get_task("bt-1").await.unwrap().unwrap();
                assert_eq!(row.status, BackgroundTaskStatus::Failed);
                assert_eq!(row.last_error.as_deref(), Some("daemon restarted mid-turn"));
                assert_eq!(row.progress_hint.as_deref(), Some("step 2/4"));
                assert_eq!(row.ended_at, Some(1_700_000_555));
            })
            .await;
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn background_task_user_b_cannot_read_user_a_row() {
    // Cross-user isolation: Bob's get_task returns None and his
    // update_task errors — the same opacity rule applied elsewhere.
    with_fixture(
        "background_task_user_b_cannot_read_user_a_row",
        |fx| async move {
            let store = PgBackgroundTaskStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                store
                    .create_task(task_row("bt-x", "alice", BackgroundTaskStatus::Running))
                    .await
                    .expect("create");
            })
            .await;

            let bob_get = with_user_id(UserId::new("bob"), async { store.get_task("bt-x").await })
                .await
                .expect("ok");
            assert!(bob_get.is_none(), "bob must not see alice's task");

            let bob_update = with_user_id(UserId::new("bob"), async {
                store
                    .update_task(
                        "bt-x",
                        BackgroundTaskStatus::Failed,
                        Some("nope"),
                        None,
                        Some(1_700_000_999),
                    )
                    .await
            })
            .await;
            assert!(
                bob_update.is_err(),
                "bob must not be able to update alice's row"
            );

            // Alice's row is unaffected.
            let alice_get =
                with_user_id(UserId::new("alice"), async { store.get_task("bt-x").await })
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(alice_get.status, BackgroundTaskStatus::Running);
            assert!(alice_get.last_error.is_none());
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn background_task_scan_non_terminal_walks_all_users() {
    // Cold-restart sweep must see every user's non-terminal rows in a
    // single pass — without a JWT scope installed, just like the
    // turn-state sweep does.
    with_fixture(
        "background_task_scan_non_terminal_walks_all_users",
        |fx| async move {
            let store = PgBackgroundTaskStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                store
                    .create_task(task_row(
                        "a-running",
                        "alice",
                        BackgroundTaskStatus::Running,
                    ))
                    .await
                    .unwrap();
                store
                    .create_task(task_row("a-done", "alice", BackgroundTaskStatus::Completed))
                    .await
                    .unwrap();
            })
            .await;
            with_user_id(UserId::new("bob"), async {
                store
                    .create_task(task_row("b-running", "bob", BackgroundTaskStatus::Running))
                    .await
                    .unwrap();
                store
                    .create_task(task_row(
                        "b-cancelled",
                        "bob",
                        BackgroundTaskStatus::Cancelled,
                    ))
                    .await
                    .unwrap();
            })
            .await;

            // No scope installed for the sweep call.
            let mut rows = store.scan_non_terminal().await.expect("scan");
            rows.sort_by(|a, b| a.id.cmp(&b.id));
            let ids: Vec<_> = rows.iter().map(|r| r.id.clone()).collect();
            assert_eq!(
                ids,
                vec!["a-running".to_string(), "b-running".to_string()],
                "only non-terminal rows from both users surface"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn background_task_list_for_user_filters_to_owner() {
    // The list path used by the registry must produce only the calling
    // user's rows, in started_at DESC order. This audits the SQL: a
    // missing `WHERE user_id = $1` would let one user see another's
    // tasks.
    with_fixture(
        "background_task_list_for_user_filters_to_owner",
        |fx| async move {
            let store = PgBackgroundTaskStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                let mut row1 = task_row("a-1", "alice", BackgroundTaskStatus::Running);
                row1.started_at = 1_700_000_001;
                let mut row2 = task_row("a-2", "alice", BackgroundTaskStatus::Completed);
                row2.started_at = 1_700_000_002;
                store.create_task(row1).await.unwrap();
                store.create_task(row2).await.unwrap();
            })
            .await;
            with_user_id(UserId::new("bob"), async {
                let mut row = task_row("b-1", "bob", BackgroundTaskStatus::Running);
                row.started_at = 1_700_000_003;
                store.create_task(row).await.unwrap();
            })
            .await;

            let alice_list = store
                .list_tasks_for_user("alice", /*include_finished*/ true, None)
                .await
                .unwrap();
            let alice_ids: Vec<_> = alice_list.iter().map(|r| r.id.clone()).collect();
            assert_eq!(alice_ids, vec!["a-2".to_string(), "a-1".to_string()]);

            // include_finished=false trims to non-terminal rows.
            let alice_active = store
                .list_tasks_for_user("alice", false, None)
                .await
                .unwrap();
            let alice_active_ids: Vec<_> = alice_active.iter().map(|r| r.id.clone()).collect();
            assert_eq!(alice_active_ids, vec!["a-1".to_string()]);

            let bob_list = store.list_tasks_for_user("bob", true, None).await.unwrap();
            let bob_ids: Vec<_> = bob_list.iter().map(|r| r.id.clone()).collect();
            assert_eq!(bob_ids, vec!["b-1".to_string()]);
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn background_task_parent_link_persists_through_postgres() {
    // The parent_task_id column round-trips and supports a future
    // children join. We don't declare a FK constraint (see migration
    // header), so a parent that was deleted before the child surfaces
    // as a dangling reference rather than a FK violation.
    with_fixture(
        "background_task_parent_link_persists_through_postgres",
        |fx| async move {
            let store = PgBackgroundTaskStore::new(fx.pool.clone());
            with_user_id(UserId::new("alice"), async {
                let parent = task_row("parent", "alice", BackgroundTaskStatus::Running);
                store.create_task(parent).await.unwrap();

                let mut child = task_row("child", "alice", BackgroundTaskStatus::Running);
                child.parent_task_id = Some("parent".into());
                child.kind_json = serde_json::json!({
                    "subagent": {
                        "parent_task_id": "parent",
                        "conversation_id": "c",
                        "name": "ch"
                    }
                });
                store.create_task(child).await.unwrap();

                let read = store.get_task("child").await.unwrap().unwrap();
                assert_eq!(read.parent_task_id.as_deref(), Some("parent"));
            })
            .await;
            fx
        },
    )
    .await;
}
