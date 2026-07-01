//! Integration tests for the #434 Postgres Row-Level Security backstop.
//!
//! The AST grafter (#141, see `database_query_user_id_scoping.rs`) rewrites
//! every SELECT the LLM-facing `execute_database_query` tool runs to append
//! `user_id = $caller`. That rewriter is the *only* text-level defense. This
//! suite pins the defense-in-depth layer underneath it: migration
//! `029_rls_backstop.sql` enables RLS on every user-scoped table and the read
//! path drops into the un-privileged `adele_query` role with `app.user_id`
//! pinned, so Postgres itself filters rows to the caller — regardless of what
//! the SQL text says.
//!
//! Acceptance criteria (from the issue), each a named test below:
//! - `rls_blocks_cross_tenant_even_without_graft` — a query with grafting
//!   deliberately bypassed still returns zero foreign rows.
//! - `rls_role_cannot_bypass` — the tool role lacks BYPASSRLS (and superuser).
//! - `trusted_owner_still_sees_all_users` — trusted app queries unaffected.
//!
//! Plus two guards beyond the issue's list:
//! - `read_path_engages_rls_end_to_end` — the real `execute_database_query`
//!   read path returns only the caller's rows with graft AND RLS both live.
//! - `rls_enabled_on_every_user_scoped_table` — drift guard: every table with
//!   a `user_id` column has RLS on and a policy, so a future table can't be
//!   added user-scoped-but-unprotected.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`).

mod support;

use desktop_assistant_core::domain::{Conversation, Message, Role};
use desktop_assistant_core::ports::store::ConversationStore;
use desktop_assistant_storage::{
    PgConversationStore, TOOL_QUERY_ROLE, UserId, execute_database_query, with_user_id,
};
use sqlx::{PgPool, Row};

use support::DbFixture;

/// Seed one conversation each for `alice` and `bob`, written as the owner
/// role (RLS-exempt), so both rows exist for the isolation assertions.
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

/// Build a fixture with the tool role granted on its private schema, or
/// pass-skip when `TEST_DATABASE_URL` is unset.
async fn fixture(prefix: &str) -> Option<DbFixture> {
    let fx = DbFixture::try_new(prefix).await?;
    // Production tables live in `public` (granted by migration 029); this
    // suite's live in a private schema, so grant the tool role there too.
    support::grant_tool_role_on_schema(&fx.pool, fx.schema()).await;
    Some(fx)
}

#[tokio::test]
async fn rls_blocks_cross_tenant_even_without_graft() {
    let Some(fx) = fixture("rls_nograft").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; rls_blocks_cross_tenant_even_without_graft");
        return;
    };
    seed_two_users(&fx.pool).await;

    // Replicate the read path's transaction setup EXACTLY, but run a raw
    // `SELECT` with no `user_id` predicate at all — i.e. grafting is
    // bypassed. If RLS is the real backstop, Postgres must still return only
    // alice's rows.
    let mut tx = fx.pool.begin().await.expect("begin");
    sqlx::query("SET TRANSACTION READ ONLY")
        .execute(&mut *tx)
        .await
        .expect("read only");
    sqlx::query("SELECT set_config('app.user_id', $1, true)")
        .bind("alice")
        .execute(&mut *tx)
        .await
        .expect("pin app.user_id");
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "SET LOCAL ROLE {TOOL_QUERY_ROLE}"
    )))
    .execute(&mut *tx)
    .await
    .expect("assume tool role");

    let rows = sqlx::query("SELECT id, user_id FROM conversations")
        .fetch_all(&mut *tx)
        .await
        .expect("ungrafted select under RLS");
    tx.rollback().await.expect("rollback");

    let ids: Vec<String> = rows.iter().map(|r| r.get::<String, _>("id")).collect();
    let users: Vec<String> = rows.iter().map(|r| r.get::<String, _>("user_id")).collect();
    assert!(
        !ids.is_empty(),
        "alice must still see her own row — RLS should filter, not blank everything"
    );
    assert!(
        users.iter().all(|u| u == "alice"),
        "RLS must return only alice's rows even with no WHERE user_id filter; got users={users:?}"
    );
    assert!(
        !ids.contains(&"conv-bob".to_string()),
        "bob's conversation leaked past RLS: {ids:?}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn rls_role_cannot_bypass() {
    let Some(fx) = fixture("rls_role").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; rls_role_cannot_bypass");
        return;
    };

    let row =
        sqlx::query("SELECT rolsuper, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname = $1")
            .bind(TOOL_QUERY_ROLE)
            .fetch_one(&fx.pool)
            .await
            .expect("adele_query role must exist after migration 029");

    assert!(
        !row.get::<bool, _>("rolbypassrls"),
        "the tool role MUST NOT have BYPASSRLS — that would defeat the backstop"
    );
    assert!(
        !row.get::<bool, _>("rolsuper"),
        "the tool role MUST NOT be a superuser (superusers bypass RLS)"
    );
    assert!(
        !row.get::<bool, _>("rolcanlogin"),
        "the tool role is entered via SET ROLE only; it must not be able to log in"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn trusted_owner_still_sees_all_users() {
    let Some(fx) = fixture("rls_owner").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; trusted_owner_still_sees_all_users");
        return;
    };
    seed_two_users(&fx.pool).await;

    // The owner role (what the daemon's own code paths use) is exempt from
    // non-FORCE RLS, so a plain owner query still sees every user's rows —
    // trusted code is not constrained by the backstop.
    let count: (i64,) = sqlx::query_as("SELECT count(*) FROM conversations")
        .fetch_one(&fx.pool)
        .await
        .expect("owner count");
    assert_eq!(
        count.0, 2,
        "the trusted owner role must see BOTH users' rows (RLS is non-FORCE)"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn read_path_engages_rls_end_to_end() {
    let Some(fx) = fixture("rls_e2e").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; read_path_engages_rls_end_to_end");
        return;
    };
    seed_two_users(&fx.pool).await;

    // The real tool path, as alice: graft AND RLS both live. Only alice's
    // row comes back, and the read path succeeds under the restricted role
    // (proving the grants + role-switch are wired correctly end to end).
    let result = with_user_id(UserId::new("alice"), async {
        execute_database_query(&fx.pool, "SELECT id, user_id FROM conversations", 100)
            .await
            .expect("read under tool role + RLS")
    })
    .await;

    let rows = result["rows"].as_array().expect("rows array");
    assert_eq!(
        rows.len(),
        1,
        "alice must see exactly her own row: {result}"
    );
    assert_eq!(rows[0][0], "conv-alice", "wrong row returned: {result}");
    assert_eq!(rows[0][1], "alice");

    fx.cleanup().await;
}

#[tokio::test]
async fn read_path_runs_as_the_restricted_tool_role() {
    let Some(fx) = fixture("rls_whoami").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; read_path_runs_as_the_restricted_tool_role");
        return;
    };

    // Direct proof that `execute_read` engages the role switch (not just the
    // graft): `SELECT current_user` passes through ungrafted, so whatever it
    // returns IS the role the read path ran under. If the `SET LOCAL ROLE`
    // were dropped, this would report the owner role and the assert fails —
    // the one signal that catches a regression the redundant graft would hide.
    let result = with_user_id(UserId::new("alice"), async {
        execute_database_query(&fx.pool, "SELECT current_user AS who", 100)
            .await
            .expect("select current_user")
    })
    .await;

    let rows = result["rows"].as_array().expect("rows array");
    assert_eq!(
        rows[0][0], TOOL_QUERY_ROLE,
        "read path must run under the restricted `{TOOL_QUERY_ROLE}` role, got: {result}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn rls_enabled_on_every_user_scoped_table() {
    let Some(fx) = fixture("rls_drift").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; rls_enabled_on_every_user_scoped_table");
        return;
    };

    // Every base table carrying a `user_id` column must have RLS enabled AND
    // at least one policy. Derived from the live catalog (not a hand list),
    // so a future migration that adds a user-scoped table without protecting
    // it fails here — the same drift-guard philosophy as #447.
    let rows = sqlx::query(
        "SELECT c.relname AS table_name, c.relrowsecurity AS rls_on, count(p.polname) AS policies
         FROM pg_class c
         JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = $1
         JOIN information_schema.columns col
             ON col.table_schema = $1 AND col.table_name = c.relname
            AND col.column_name = 'user_id'
         LEFT JOIN pg_policy p ON p.polrelid = c.oid
         WHERE c.relkind = 'r'
         GROUP BY c.relname, c.relrowsecurity
         ORDER BY c.relname",
    )
    .bind(fx.schema())
    .fetch_all(&fx.pool)
    .await
    .expect("catalog query");

    assert!(
        !rows.is_empty(),
        "expected user-scoped tables in the schema; found none (fixture broken?)"
    );
    let mut unprotected = Vec::new();
    for r in &rows {
        let name: String = r.get("table_name");
        let rls_on: bool = r.get("rls_on");
        let policies: i64 = r.get("policies");
        if !rls_on || policies == 0 {
            unprotected.push(format!("{name} (rls_on={rls_on}, policies={policies})"));
        }
    }
    assert!(
        unprotected.is_empty(),
        "user-scoped tables missing RLS/policy — add them to migration 029: {unprotected:?}"
    );

    fx.cleanup().await;
}
