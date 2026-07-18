//! Integration tests for issue #501: `db_query` cross-tenant isolation on a
//! *two-tenant* database.
//!
//! During the 2026-07-15 tool-search outage, db_query tenant isolation was
//! only ever verified fail-closed on a *single*-user DB. The read path
//! (`crates/storage/src/database.rs` `execute_read` — READ ONLY tx,
//! `set_config('app.user_id', $1, true)`, `SET LOCAL ROLE adele_query`, always
//! rollback) has AST-graft (#141) *and* Postgres-RLS (#434) defense in depth,
//! but nothing seeded two real tenants and proved that tenant A cannot read
//! tenant B's personal rows while both can still read the shared catalogs.
//!
//! This suite closes that gap. Every test drives the *real* LLM path
//! (`BuiltinToolService::execute_tool("builtin_db_query", …)`, wired exactly as
//! `crates/daemon/src/main.rs` wires it) under `with_user_id(...)` — so a
//! regression anywhere in the grafter, the role switch, or the RLS policies
//! surfaces here. The assertions check *emptiness of the other tenant's rows*,
//! not merely that the call returned `Ok`: a genuine cross-tenant leak makes
//! these fail.
//!
//! This is verification only — it adds no isolation mechanism. See
//! `rls_backstop.rs` for the by-hand RLS proof and `audit_user_id_scoping.rs`
//! for the static AST-scan coverage.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips (loudly, via `support`) when unset.
//! Run against an ephemeral Postgres with:
//!
//! ```sh
//! just test-db -p desktop-assistant-storage --test db_query_two_tenant_isolation
//! ```

mod support;

use std::sync::Arc;

use desktop_assistant_core::domain::{Conversation, Message, Role};
use desktop_assistant_core::ports::database::DbQueryFn;
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_mcp_client::executor::BuiltinToolService;
use desktop_assistant_storage::{
    PgConversationStore, UserId, execute_database_query, with_user_id,
};
use sqlx::PgPool;

use support::DbFixture;

/// Build a fixture with the #434 tool role granted on its private schema, or
/// pass-skip when `TEST_DATABASE_URL` is unset. `provision_tool_role` MUST run
/// *after* `run_migrations` (done inside `DbFixture::try_new`) or every grafted
/// SELECT hits "permission denied for table" once the read path drops into the
/// un-privileged `adele_query` role.
async fn fixture(prefix: &str) -> Option<DbFixture> {
    let fx = DbFixture::try_new(prefix).await?;
    support::provision_tool_role(&fx.pool, fx.schema()).await;
    Some(fx)
}

/// Seed one conversation each for `alice` and `bob`, written as the owner role
/// (RLS-exempt) so both rows exist for the isolation assertions.
async fn seed_two_users(pool: &PgPool) {
    let store = PgConversationStore::new(pool.clone());
    for (user, id, title, body) in [
        ("alice", "conv-alice", "alice's chat", "alice's secret"),
        ("bob", "conv-bob", "bob's chat", "bob's secret"),
    ] {
        with_user_id(UserId::new(user), async {
            let mut conv = Conversation::new(id, title);
            conv.created_at = "2026-01-01 00:00:00".to_string();
            conv.updated_at = "2026-01-01 00:00:00".to_string();
            conv.messages.push(Message::new(Role::User, body));
            store.create(conv).await.expect("seed create");
        })
        .await;
    }
}

/// Drive the production db_query tool path over `pool`, running `query` as
/// `user` through `BuiltinToolService::execute_tool("builtin_db_query", …)` —
/// the verbatim LLM tool-call path — and return the decoded `rows` array.
///
/// Panics with a legible message on any transport-level failure (a "permission
/// denied" under the tool role, a malformed response) so the assertions in each
/// test speak only to the *data*, not the plumbing.
async fn db_query_rows_as(pool: &PgPool, user: &str, query: &str) -> Vec<serde_json::Value> {
    let pool_for_fn = pool.clone();
    let query_fn: DbQueryFn = Arc::new(move |sql, limit| {
        let pool = pool_for_fn.clone();
        Box::pin(async move { execute_database_query(&pool, &sql, limit).await })
    });
    let service = BuiltinToolService::new().with_database(query_fn);

    // The transport handler installs `with_user_id(...)` around the LLM's
    // tool-call dispatch in production; do the same here.
    let response = with_user_id(UserId::new(user), async {
        service
            .execute_tool("builtin_db_query", serde_json::json!({ "query": query }))
            .await
    })
    .await
    .unwrap_or_else(|e| panic!("db_query tool failed for {user}: {e:?}"));

    let json: serde_json::Value = serde_json::from_str(&response).expect("tool response is JSON");
    assert_eq!(
        json["ok"],
        serde_json::json!(true),
        "tool did not report ok for {user}: {json}"
    );
    json["result"]["rows"]
        .as_array()
        .expect("result.rows is an array")
        .clone()
}

/// Extract the first column of each row as a string, for terse assertions.
fn first_col(rows: &[serde_json::Value]) -> Vec<String> {
    rows.iter()
        .map(|r| r[0].as_str().unwrap_or("").to_string())
        .collect()
}

#[tokio::test]
async fn db_query_user_a_reads_only_own_personal_rows() {
    let Some(fx) = fixture("dq2t_own").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; db_query_user_a_reads_only_own_personal_rows");
        return;
    };
    seed_two_users(&fx.pool).await;

    // Alice asks for every conversation. Graft + RLS both narrow the read to
    // her own rows before any data leaves the pool.
    let rows = db_query_rows_as(
        &fx.pool,
        "alice",
        "SELECT id, title FROM conversations ORDER BY id",
    )
    .await;

    let ids = first_col(&rows);
    assert_eq!(
        ids,
        vec!["conv-alice".to_string()],
        "alice must see exactly her own conversation, got {ids:?}"
    );
    assert_eq!(rows[0][1], "alice's chat", "wrong row content: {rows:?}");
    assert!(
        !ids.iter().any(|id| id == "conv-bob"),
        "bob's conversation must never appear in alice's result: {ids:?}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn db_query_user_b_cannot_read_user_a_personal_rows() {
    let Some(fx) = fixture("dq2t_crosstenant").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; db_query_user_b_cannot_read_user_a_personal_rows"
        );
        return;
    };
    seed_two_users(&fx.pool).await;

    // Bob runs the *identical* unscoped query alice ran. A real cross-tenant
    // leak would surface `conv-alice`; the isolation guarantee means it must
    // not. Assert the alice-slice is empty, not merely that the call is Ok.
    let rows = db_query_rows_as(
        &fx.pool,
        "bob",
        "SELECT id, title FROM conversations ORDER BY id",
    )
    .await;

    let ids = first_col(&rows);
    let alice_leak: Vec<&String> = ids
        .iter()
        .filter(|id| id.starts_with("conv-alice"))
        .collect();
    assert!(
        alice_leak.is_empty(),
        "bob must NEVER read alice's rows; leaked: {alice_leak:?} (full set {ids:?})"
    );
    assert_eq!(
        ids,
        vec!["conv-bob".to_string()],
        "bob must see exactly his own conversation, got {ids:?}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn db_query_explicit_other_user_id_predicate_returns_zero_rows() {
    let Some(fx) = fixture("dq2t_explicit").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; \
             db_query_explicit_other_user_id_predicate_returns_zero_rows"
        );
        return;
    };
    seed_two_users(&fx.pool).await;

    // The hostile case: bob knows alice's user_id (a JWT `sub`, visible in
    // tokens) and spells it out. The grafter AND's `bob.user_id = $caller`
    // onto his predicate, and RLS pins `app.user_id = 'bob'` underneath, so
    // the intersection {rows where user_id='alice'} ∩ {rows visible to bob}
    // is empty.
    let rows = db_query_rows_as(
        &fx.pool,
        "bob",
        "SELECT id, title FROM conversations WHERE user_id = 'alice'",
    )
    .await;

    assert!(
        rows.is_empty(),
        "an explicit `WHERE user_id='alice'` run as bob must return zero rows, got {rows:?}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn db_query_shared_catalog_tool_definitions_readable_by_both_tenants() {
    let Some(fx) = fixture("dq2t_catalog").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; \
             db_query_shared_catalog_tool_definitions_readable_by_both_tenants"
        );
        return;
    };

    // `tool_definitions` is the system-wide MCP tool registry (a GLOBAL table:
    // no `user_id` column, not RLS-protected, not grafted). Seed it as the
    // owner, then prove BOTH tenants read the same global rows through the tool
    // path — isolation must not blank the shared catalog.
    for (name, desc) in [
        ("alpha_tool", "first shared tool"),
        ("beta_tool", "second shared tool"),
    ] {
        sqlx::query(
            "INSERT INTO tool_definitions (name, description, parameters, source) \
             VALUES ($1, $2, '{}'::jsonb, 'test')",
        )
        .bind(name)
        .bind(desc)
        .execute(&fx.pool)
        .await
        .expect("seed tool_definitions");
    }

    let expected = vec!["alpha_tool".to_string(), "beta_tool".to_string()];

    let alice_names = first_col(
        &db_query_rows_as(
            &fx.pool,
            "alice",
            "SELECT name FROM tool_definitions ORDER BY name",
        )
        .await,
    );
    let bob_names = first_col(
        &db_query_rows_as(
            &fx.pool,
            "bob",
            "SELECT name FROM tool_definitions ORDER BY name",
        )
        .await,
    );

    assert_eq!(
        alice_names, expected,
        "alice must read the full shared tool catalog, got {alice_names:?}"
    );
    assert_eq!(
        bob_names, expected,
        "bob must read the full shared tool catalog, got {bob_names:?}"
    );
    assert_eq!(
        alice_names, bob_names,
        "the shared catalog must be identical across tenants (no per-user filtering)"
    );

    fx.cleanup().await;
}
