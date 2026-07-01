//! Integration coverage for row-level storage error/lifecycle branches
//! (issue #443).
//!
//! Each helper below guards a silent-corruption path that only the happy path
//! exercised before this suite:
//!
//! - duplicate-id INSERTs (`turn_state`, `background_tasks`) must ERROR, not
//!   silently drop the row (a regression to `ON CONFLICT DO NOTHING`);
//! - the summary lifecycle (`create_summary` stamping `summary_id` on a range,
//!   `expand_summary` clearing it via `ON DELETE SET NULL`);
//! - `archive` opacity (foreign/missing ⇒ `NotFound`, already-archived ⇒ `Ok`);
//! - `tag_registry` embedding-dedup redirect and deprecation-chain / cycle
//!   guard;
//! - `tool_registry` upsert / hybrid search / source-scoped unregister;
//! - JSON→Postgres migration of conversations + knowledge and the empty-table
//!   probes.
//!
//! When `TEST_DATABASE_URL` is unset every test pass-skips.

mod support;

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, KnowledgeEntry, Message, Role, ToolDefinition,
};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore, ConversationStore, TurnRow,
    TurnStateJson, TurnStateStore, TurnStatus,
};
use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
use desktop_assistant_storage::embedding_backfill::BackfillEmbedFn;
use desktop_assistant_storage::tag_registry::{
    CreateTagOutcome, TagProposal, create_or_match_tag, resolve_active_name,
};
use desktop_assistant_storage::{
    PgBackgroundTaskStore, PgConversationStore, PgKnowledgeBaseStore, PgToolRegistryStore,
    PgTurnStateStore, UserId, is_conversations_table_empty, is_knowledge_base_table_empty,
    migrate_conversations, migrate_knowledge, run_migrations, with_user_id,
};
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
        let schema = format!("issue443_{}", Uuid::now_v7().simple());

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

fn task_row(id: &str, user_id: &str, status: BackgroundTaskStatus) -> BackgroundTaskRow {
    BackgroundTaskRow {
        id: id.into(),
        user_id: user_id.into(),
        kind_json: serde_json::json!({"standalone": {"name": "t", "conversation_id": "c"}}),
        status,
        parent_task_id: None,
        title: format!("title {id}"),
        last_error: None,
        progress_hint: None,
        started_at: 1_700_000_000,
        ended_at: None,
    }
}

fn conversation_with_messages(id: &str, bodies: &[&str]) -> Conversation {
    let mut conv = Conversation::new(id, "row-level test conversation");
    conv.created_at = "2026-01-01 00:00:00".to_string();
    conv.updated_at = "2026-01-01 00:00:00".to_string();
    for body in bodies {
        conv.messages.push(Message::new(Role::User, *body));
    }
    conv
}

// -- duplicate-id INSERTs must error -----------------------------------------

#[tokio::test]
async fn create_duplicate_turn_id_errors() {
    // A second `create_turn` with the same id must ERROR ("already exists"), not
    // silently no-op. This pins the unique-violation branch (turn_state.rs:82).
    //
    // MUTATION: rewriting the INSERT to `ON CONFLICT (id) DO NOTHING` makes the
    // second create return Ok → RED.
    with_fixture("create_duplicate_turn_id_errors", |fx| async move {
        let store = PgTurnStateStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            store
                .create_turn(turn_row("t-dup", "alice", "c", TurnStatus::PendingLlm))
                .await
                .expect("first create");

            let second = store
                .create_turn(turn_row("t-dup", "alice", "c", TurnStatus::PendingLlm))
                .await;
            match second {
                Err(CoreError::Storage(msg)) => assert!(
                    msg.contains("already exists"),
                    "duplicate-turn error should name the collision, got: {msg}"
                ),
                other => panic!("expected a Storage 'already exists' error, got {other:?}"),
            }
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn create_duplicate_task_id_errors() {
    // Same contract for background tasks (background_tasks.rs:91).
    //
    // MUTATION: `ON CONFLICT (id) DO NOTHING` → second create Ok → RED.
    with_fixture("create_duplicate_task_id_errors", |fx| async move {
        let store = PgBackgroundTaskStore::new(fx.pool.clone());
        with_user_id(UserId::new("alice"), async {
            store
                .create_task(task_row("bt-dup", "alice", BackgroundTaskStatus::Running))
                .await
                .expect("first create");

            let second = store
                .create_task(task_row("bt-dup", "alice", BackgroundTaskStatus::Running))
                .await;
            match second {
                Err(CoreError::Storage(msg)) => assert!(
                    msg.contains("already exists"),
                    "duplicate-task error should name the collision, got: {msg}"
                ),
                other => panic!("expected a Storage 'already exists' error, got {other:?}"),
            }
        })
        .await;
        fx
    })
    .await;
}

// -- summary lifecycle -------------------------------------------------------

#[tokio::test]
async fn create_summary_stamps_range_then_get_returns_it() {
    // `create_summary` inserts a summary row AND stamps `summary_id` on exactly
    // the messages in `BETWEEN start AND end`. `get` must then return the
    // summary and show the stamp on the in-range messages only.
    //
    // MUTATION: narrowing/removing `ordinal BETWEEN $4 AND $5` leaves the
    // in-range messages unstamped (summary_id stays NULL) → RED.
    with_fixture(
        "create_summary_stamps_range_then_get_returns_it",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let cid = ConversationId::from("c-sum");

            let sid = with_user_id(UserId::new("alice"), async {
                store
                    .create(conversation_with_messages(
                        "c-sum",
                        &["m0", "m1", "m2", "m3"],
                    ))
                    .await
                    .expect("create");
                store
                    .create_summary(&cid, "range summary".to_string(), 1, 2)
                    .await
                    .expect("create_summary")
            })
            .await;

            let got = with_user_id(UserId::new("alice"), async { store.get(&cid).await })
                .await
                .expect("get");

            assert!(
                got.summaries
                    .iter()
                    .any(|s| s.id == sid && s.summary == "range summary"),
                "the created summary must round-trip through get(); summaries = {:?}",
                got.summaries
            );
            assert_eq!(
                got.messages[1].summary_id.as_deref(),
                Some(sid.as_str()),
                "message at ordinal 1 must be stamped with the summary id"
            );
            assert_eq!(
                got.messages[2].summary_id.as_deref(),
                Some(sid.as_str()),
                "message at ordinal 2 must be stamped with the summary id"
            );
            assert!(
                got.messages[0].summary_id.is_none(),
                "ordinal 0 is outside the range and must NOT be stamped"
            );
            assert!(
                got.messages[3].summary_id.is_none(),
                "ordinal 3 is outside the range and must NOT be stamped"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn expand_summary_clears_summary_id() {
    // `expand_summary` deletes the summary row; the `ON DELETE SET NULL` FK on
    // `messages.summary_id` then clears every reference. After expand, get()
    // shows no summaries and no stamped messages.
    //
    // MUTATION: pointing the DELETE at a non-matching id (so nothing is
    // deleted) leaves the messages stamped → RED.
    with_fixture("expand_summary_clears_summary_id", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let cid = ConversationId::from("c-exp");

        with_user_id(UserId::new("alice"), async {
            store
                .create(conversation_with_messages("c-exp", &["m0", "m1", "m2"]))
                .await
                .expect("create");
            let sid = store
                .create_summary(&cid, "collapse me".to_string(), 0, 1)
                .await
                .expect("create_summary");

            // Precondition: the stamp landed.
            let before = store.get(&cid).await.expect("get before");
            assert_eq!(before.messages[0].summary_id.as_deref(), Some(sid.as_str()));

            store.expand_summary(&sid).await.expect("expand_summary");

            let after = store.get(&cid).await.expect("get after");
            assert!(
                after.summaries.is_empty(),
                "expand_summary must remove the summary row; got {:?}",
                after.summaries
            );
            assert!(
                after.messages.iter().all(|m| m.summary_id.is_none()),
                "ON DELETE SET NULL must clear every message's summary_id after \
                 expand; got {:?}",
                after
                    .messages
                    .iter()
                    .map(|m| &m.summary_id)
                    .collect::<Vec<_>>()
            );
        })
        .await;
        fx
    })
    .await;
}

// -- archive opacity ---------------------------------------------------------

#[tokio::test]
async fn archive_foreign_conversation_is_not_found() {
    // Archiving a conversation you don't own returns `ConversationNotFound`
    // (the `rows_affected == 0` probe is itself user-scoped so it can't leak
    // existence), and the owner's row stays unarchived.
    //
    // MUTATION: dropping `user_id = $1` from the existence probe makes Bob's
    // probe find Alice's row → archive returns Ok → RED.
    with_fixture(
        "archive_foreign_conversation_is_not_found",
        |fx| async move {
            let store = PgConversationStore::new(fx.pool.clone());
            let cid = ConversationId::from("c-arch");

            with_user_id(UserId::new("alice"), async {
                store
                    .create(conversation_with_messages("c-arch", &["hello"]))
                    .await
                    .expect("alice create");
            })
            .await;

            let bob = with_user_id(UserId::new("bob"), async { store.archive(&cid).await }).await;
            match bob {
                Err(CoreError::ConversationNotFound(id)) => assert_eq!(id, "c-arch"),
                other => panic!("expected ConversationNotFound, got {other:?}"),
            }

            // Alice's conversation is still active (bob's attempt did nothing).
            let alice = with_user_id(UserId::new("alice"), async { store.get(&cid).await })
                .await
                .expect("alice get");
            assert!(
                alice.archived_at.is_none(),
                "alice's conversation must remain unarchived after bob's failed \
             archive"
            );
            fx
        },
    )
    .await;
}

#[tokio::test]
async fn double_archive_is_ok() {
    // Archiving an already-archived conversation the caller owns is a no-op that
    // returns Ok — the `rows_affected == 0` probe distinguishes "already
    // archived" (exists ⇒ Ok) from "missing/foreign" (absent ⇒ NotFound).
    //
    // MUTATION: flipping the probe guard from `if exists.is_none()` to
    // `if exists.is_some()` makes the second archive return NotFound → RED.
    with_fixture("double_archive_is_ok", |fx| async move {
        let store = PgConversationStore::new(fx.pool.clone());
        let cid = ConversationId::from("c-dbl");

        with_user_id(UserId::new("alice"), async {
            store
                .create(conversation_with_messages("c-dbl", &["hello"]))
                .await
                .expect("create");

            store.archive(&cid).await.expect("first archive");
            store
                .archive(&cid)
                .await
                .expect("second archive of own already-archived conversation is Ok");

            // Still archived.
            let got = store.get(&cid).await.expect("get");
            assert!(
                got.archived_at.is_some(),
                "the conversation must remain archived after the idempotent \
                 second archive"
            );
        })
        .await;
        fx
    })
    .await;
}

// -- tag_registry ------------------------------------------------------------

/// An embedding function that always returns a fixed vector, so a proposed tag
/// deterministically lands at cosine distance 0 from an already-stored one.
/// Follows the boxed-future idiom of `maintenance_service::build_embed_fn`.
fn fixed_embed_fn(vector: Vec<f32>) -> BackfillEmbedFn {
    Box::new(move |texts| {
        let vector = vector.clone();
        Box::pin(async move { Ok(texts.iter().map(|_| vector.clone()).collect()) })
    })
}

#[tokio::test]
async fn tag_dedup_redirects_to_canonical() {
    // A newly proposed tag whose embedding is within
    // `TAG_DEDUP_DISTANCE_THRESHOLD` of an existing active tag is redirected to
    // that canonical tag rather than inserted (the pre-flight similarity check).
    //
    // MUTATION: flipping `distance < TAG_DEDUP_DISTANCE_THRESHOLD` to `>` (or
    // removing the redirect) makes the near-duplicate insert as `Created` → RED.
    with_fixture("tag_dedup_redirects_to_canonical", |fx| async move {
        let embed_fn = fixed_embed_fn(vec![1.0, 0.0, 0.0]);

        with_user_id(UserId::new("alice"), async {
            let created = create_or_match_tag(
                &fx.pool,
                &embed_fn,
                "test-model",
                TagProposal {
                    name: "project".into(),
                    description: "a unit of work".into(),
                    examples: vec![],
                    distinguish_from: vec![],
                },
            )
            .await
            .expect("create canonical");
            assert!(
                matches!(created, CreateTagOutcome::Created(ref t) if t.name == "project"),
                "the first tag should be freshly created, got {created:?}"
            );

            // Different normalized name (so the exact-match short-circuit does
            // NOT fire) but an identical embedding ⇒ redirect on similarity.
            let redirected = create_or_match_tag(
                &fx.pool,
                &embed_fn,
                "test-model",
                TagProposal {
                    name: "projects".into(),
                    description: "grouping of tasks".into(),
                    examples: vec![],
                    distinguish_from: vec![],
                },
            )
            .await
            .expect("propose near-duplicate");
            match redirected {
                CreateTagOutcome::RedirectedTo { existing, .. } => assert_eq!(
                    existing.name, "project",
                    "the near-duplicate must redirect to the canonical tag"
                ),
                other => panic!("expected RedirectedTo, got {other:?}"),
            }
        })
        .await;
        fx
    })
    .await;
}

#[tokio::test]
async fn resolve_active_name_follows_deprecation_without_cycling() {
    // `resolve_active_name` walks the `deprecated_for_tag` chain to its terminal
    // active tag, and the loop bound catches a cycle instead of hanging.
    //
    // MUTATION: replacing `current = next` with `return Ok(Some(current))` stops
    // following the chain so `resolve("old")` returns "old" → RED.
    with_fixture(
        "resolve_active_name_follows_deprecation_without_cycling",
        |fx| async move {
            // old -> mid -> new (active). Insert targets before referrers so the
            // (user_id, deprecated_for_tag) FK is satisfied.
            for (name, dep) in [("new", None), ("mid", Some("new")), ("old", Some("mid"))] {
                sqlx::query(
                    "INSERT INTO tag_registry (user_id, name, description, deprecated_for_tag) \
                     VALUES ('alice', $1, 'd', $2)",
                )
                .bind(name)
                .bind(dep)
                .execute(&fx.pool)
                .await
                .expect("insert chain tag");
            }

            with_user_id(UserId::new("alice"), async {
                let resolved = resolve_active_name(&fx.pool, "old")
                    .await
                    .expect("resolve old");
                assert_eq!(
                    resolved.as_deref(),
                    Some("new"),
                    "resolve must follow old -> mid -> new"
                );
                let terminal = resolve_active_name(&fx.pool, "new")
                    .await
                    .expect("resolve new");
                assert_eq!(
                    terminal.as_deref(),
                    Some("new"),
                    "active tag resolves to itself"
                );
                let missing = resolve_active_name(&fx.pool, "ghost")
                    .await
                    .expect("resolve missing");
                assert_eq!(missing, None, "an unknown tag resolves to None");
            })
            .await;

            // Cycle: a -> b -> a. Insert both, then wire the pointers.
            for name in ["cyc-a", "cyc-b"] {
                sqlx::query(
                    "INSERT INTO tag_registry (user_id, name, description) \
                     VALUES ('alice', $1, 'd')",
                )
                .bind(name)
                .execute(&fx.pool)
                .await
                .expect("insert cycle tag");
            }
            sqlx::query(
                "UPDATE tag_registry SET deprecated_for_tag = 'cyc-b' \
                 WHERE user_id = 'alice' AND name = 'cyc-a'",
            )
            .execute(&fx.pool)
            .await
            .expect("wire a->b");
            sqlx::query(
                "UPDATE tag_registry SET deprecated_for_tag = 'cyc-a' \
                 WHERE user_id = 'alice' AND name = 'cyc-b'",
            )
            .execute(&fx.pool)
            .await
            .expect("wire b->a");

            let cyc = with_user_id(UserId::new("alice"), async {
                resolve_active_name(&fx.pool, "cyc-a").await
            })
            .await;
            assert!(
                matches!(cyc, Err(CoreError::Storage(_))),
                "a deprecation cycle must be rejected by the depth guard, got {cyc:?}"
            );
            fx
        },
    )
    .await;
}

// -- tool_registry -----------------------------------------------------------

fn tool(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition::new(name, description, serde_json::json!({"type": "object"}))
}

#[tokio::test]
async fn tool_registry_upsert_and_search() {
    // `register_tools` upserts by name; `search_tools` runs the hybrid
    // (vector + FTS) path when given a non-empty embedding.
    //
    // MUTATION (upsert): changing `ON CONFLICT (name) DO UPDATE` to
    // `DO NOTHING` leaves the stale description after re-register → RED.
    // MUTATION (search): trimming the final `LIMIT $4` to `LIMIT 0` returns no
    // hits → RED.
    with_fixture("tool_registry_upsert_and_search", |fx| async move {
        let store = PgToolRegistryStore::new(fx.pool.clone());

        store
            .register_tools(
                vec![
                    tool("get_weather", "Get the weather forecast for a city"),
                    tool("get_time", "Return the current time in a timezone"),
                ],
                "srcA",
                false,
                vec![
                    Some(vec![vec![1.0, 0.0, 0.0]]),
                    Some(vec![vec![0.0, 1.0, 0.0]]),
                ],
                Some("test-model".to_string()),
            )
            .await
            .expect("register");

        // Hybrid search: embedding points at get_weather AND the FTS query
        // matches its description.
        let hits = store
            .search_tools("weather forecast", vec![1.0, 0.0, 0.0], 10)
            .await
            .expect("search");
        assert!(
            hits.iter().any(|t| t.name == "get_weather"),
            "hybrid search must surface the matching tool; got {:?}",
            hits.iter().map(|t| &t.name).collect::<Vec<_>>()
        );

        // Upsert: re-register the same name with a new description.
        store
            .register_tools(
                vec![tool("get_weather", "Weather now updated description")],
                "srcA",
                false,
                vec![Some(vec![vec![1.0, 0.0, 0.0]])],
                Some("test-model".to_string()),
            )
            .await
            .expect("re-register");
        let def = store
            .tool_definition("get_weather")
            .await
            .expect("lookup")
            .expect("still present");
        assert_eq!(
            def.description, "Weather now updated description",
            "the upsert must replace the description in place, not insert a dup"
        );
        fx
    })
    .await;
}

#[tokio::test]
async fn unregister_source_removes_only_that_source() {
    // `unregister_source` deletes every tool from a given source and leaves
    // other sources intact.
    //
    // MUTATION: flipping `WHERE source = $1` to `source != $1` deletes the OTHER
    // source's tools (and keeps the target's) → RED.
    with_fixture(
        "unregister_source_removes_only_that_source",
        |fx| async move {
            let store = PgToolRegistryStore::new(fx.pool.clone());

            store
                .register_tools(
                    vec![tool("a_tool", "tool from source A")],
                    "srcA",
                    false,
                    vec![None],
                    None,
                )
                .await
                .expect("register srcA");
            store
                .register_tools(
                    vec![tool("b_tool", "tool from source B")],
                    "srcB",
                    false,
                    vec![None],
                    None,
                )
                .await
                .expect("register srcB");

            store.unregister_source("srcA").await.expect("unregister");

            assert!(
                store
                    .tool_definition("a_tool")
                    .await
                    .expect("lookup a")
                    .is_none(),
                "srcA's tool must be gone after unregister"
            );
            assert!(
                store
                    .tool_definition("b_tool")
                    .await
                    .expect("lookup b")
                    .is_some(),
                "srcB's tool must be untouched by unregistering srcA"
            );
            fx
        },
    )
    .await;
}

// -- JSON migration ----------------------------------------------------------

#[tokio::test]
async fn migrate_json_imports_conversations_and_knowledge() {
    // End-to-end JSON→Postgres migration: conversations (skipping empty ones),
    // preferences, and factual memory, plus the empty-table probes that gate
    // whether migration should run at all.
    //
    // MUTATION: removing the `if conv.messages.is_empty() { continue; }` skip
    // makes migrate_conversations count the empty conversation too (2, not 1)
    // → RED.
    with_fixture(
        "migrate_json_imports_conversations_and_knowledge",
        |fx| async move {
            // Empty-table probes are true on a fresh schema.
            assert!(
                is_conversations_table_empty(&fx.pool).await,
                "conversations table should start empty"
            );
            assert!(
                is_knowledge_base_table_empty(&fx.pool).await,
                "knowledge_base table should start empty"
            );

            let dir = std::env::temp_dir().join(format!("mig-{}", Uuid::now_v7().simple()));
            std::fs::create_dir_all(&dir).expect("create temp dir");
            let conv_path = dir.join("conversations.json");
            let prefs_path = dir.join("preferences.json");
            let mem_path = dir.join("memory.json");

            // One conversation with messages (migrated) + one empty (skipped).
            let convs = vec![
                conversation_with_messages("mig-conv-1", &["hello from json"]),
                Conversation::new("mig-conv-2", "empty, should be skipped"),
            ];
            std::fs::write(&conv_path, serde_json::to_string(&convs).unwrap())
                .expect("write conversations json");
            std::fs::write(
                &prefs_path,
                r#"{"items":[{"key":"theme","value":"dark","updated_at":0}]}"#,
            )
            .expect("write prefs json");
            std::fs::write(
                &mem_path,
                r#"{"items":[{"id":"m1","fact":"User likes Rust","tags":["lang"],"created_at":0,"updated_at":0}]}"#,
            )
            .expect("write memory json");

            let conv_count = migrate_conversations(&conv_path, &fx.pool)
                .await
                .expect("migrate conversations");
            assert_eq!(
                conv_count, 1,
                "only the conversation WITH messages is migrated (empty one skipped)"
            );

            let kb_count = migrate_knowledge(&prefs_path, &mem_path, &fx.pool)
                .await
                .expect("migrate knowledge");
            assert_eq!(
                kb_count, 2,
                "one preference + one memory should be migrated into the KB"
            );

            // Probes now report non-empty.
            assert!(!is_conversations_table_empty(&fx.pool).await);
            assert!(!is_knowledge_base_table_empty(&fx.pool).await);

            // The migrated rows are readable (migration used the `default` user).
            let conv_store = PgConversationStore::new(fx.pool.clone());
            let got = conv_store
                .get(&ConversationId::from("mig-conv-1"))
                .await
                .expect("migrated conversation is retrievable");
            assert_eq!(got.messages.len(), 1);

            let kb = PgKnowledgeBaseStore::new(fx.pool.clone());
            let pref = kb.get("pref_theme").await.expect("get pref");
            assert!(
                pref.is_some_and(|e: KnowledgeEntry| e.content.contains("dark")),
                "the migrated preference should be a KB entry"
            );
            let mem = kb.get("m1").await.expect("get memory");
            assert!(
                mem.is_some_and(|e: KnowledgeEntry| e.content.contains("Rust")),
                "the migrated memory should be a KB entry"
            );

            // A non-existent conversations file migrates nothing.
            let none_count = migrate_conversations(&dir.join("nope.json"), &fx.pool)
                .await
                .expect("missing file is not an error");
            assert_eq!(none_count, 0, "a missing JSON file migrates zero rows");

            let _ = std::fs::remove_dir_all(&dir);
            fx
        },
    )
    .await;
}
