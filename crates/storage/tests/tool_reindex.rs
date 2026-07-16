//! DB-gated integration tests for the runtime enable/disable tool-search
//! reindex (#498).
//!
//! Runtime MCP enable/disable used to update only in-memory state, so the
//! persistent `tool_definitions` search index was written exactly once at
//! startup and hot-toggled servers were undiscoverable (or lingered as dead
//! rows) until a daemon restart. The fix injects a `ToolReindexFn` into the MCP
//! executor; the daemon's closure re-writes the whole `"mcp"` source by
//! composing `unregister_source("mcp")` + `register_tools(.., "mcp", ..)`.
//!
//! This suite pins the *storage* half of that policy: the delete-then-reinsert
//! composition the daemon closure relies on. The mcp-client half (the executor
//! firing the closure) is covered by colocated unit tests in
//! `crates/mcp-client/src/executor.rs`.
//!
//! Acceptance criteria (from the issue), each a named test below:
//! - `reindex_source_inserts_newly_enabled_server_tools` — after reindexing the
//!   `"mcp"` source with a superset, the newly-enabled server's rows are present
//!   and returned by `search_tools` (the FTS fallback path, empty embedding).
//! - `reindex_source_removes_disabled_server_tools` — after reindexing with a
//!   reduced set, the disabled server's rows are gone and the rest remain.
//! - `reindex_source_is_idempotent` — repeating the same reindex yields the same
//!   row set.
//! - `reindex_source_empty_set_clears_all_mcp_rows` — disabling the last MCP
//!   server (or a zero-tool server) reindexes with an empty set: the delete runs,
//!   the INSERT batch is empty, and no `"mcp"` row survives or stays searchable.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`).

mod support;

use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
use desktop_assistant_storage::PgToolRegistryStore;
use sqlx::Row;

use support::DbFixture;

/// Two `servera` tools; a distinct `serverb` tool whose description carries a
/// unique FTS token (`telemetry`) so a full-text search can single it out.
fn server_a_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "servera__alpha",
            "Aggregate records into summaries",
            serde_json::json!({"type": "object"}),
        ),
        ToolDefinition::new(
            "servera__beta",
            "Compute rolling averages",
            serde_json::json!({"type": "object"}),
        ),
    ]
}

fn server_b_tool() -> ToolDefinition {
    ToolDefinition::new(
        "serverb__frobnicate",
        "Frobnicate the telemetry widget",
        serde_json::json!({"type": "object"}),
    )
}

/// Reindex the whole `"mcp"` source exactly as the daemon closure does:
/// delete every `"mcp"` row, then reinsert the current set with NULL embeddings
/// (the background backfill fills vectors later).
async fn reindex_mcp(store: &PgToolRegistryStore, tools: Vec<ToolDefinition>) {
    store
        .unregister_source("mcp")
        .await
        .expect("unregister mcp source");
    let embeddings = vec![None; tools.len()];
    store
        .register_tools(tools, "mcp", false, embeddings, None)
        .await
        .expect("register mcp tools");
}

/// Names currently registered under the `"mcp"` source, sorted for determinism.
async fn mcp_source_names(fx: &DbFixture) -> Vec<String> {
    let rows = sqlx::query("SELECT name FROM tool_definitions WHERE source = $1 ORDER BY name")
        .bind("mcp")
        .fetch_all(&fx.pool)
        .await
        .expect("select mcp source names");
    rows.iter().map(|r| r.get::<String, _>("name")).collect()
}

/// Build a fixture with the tool role provisioned (mirrors `rls_backstop.rs`),
/// or pass-skip when `TEST_DATABASE_URL` is unset.
async fn fixture(prefix: &str) -> Option<DbFixture> {
    let fx = DbFixture::try_new(prefix).await?;
    support::provision_tool_role(&fx.pool, fx.schema()).await;
    Some(fx)
}

#[tokio::test]
async fn reindex_source_inserts_newly_enabled_server_tools() {
    let Some(fx) = fixture("reindex_insert").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; reindex_source_inserts_newly_enabled_server_tools"
        );
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // Startup state: only server A is enabled.
    reindex_mcp(&store, server_a_tools()).await;
    assert!(
        store
            .tool_definition("serverb__frobnicate")
            .await
            .expect("lookup before enable")
            .is_none(),
        "serverb tool must not exist before it is enabled"
    );

    // Enable server B: reindex with the superset {A + B}.
    let mut superset = server_a_tools();
    superset.push(server_b_tool());
    reindex_mcp(&store, superset).await;

    // The newly-enabled server's row is present...
    assert!(
        store
            .tool_definition("serverb__frobnicate")
            .await
            .expect("lookup after enable")
            .is_some(),
        "serverb tool must be indexed after enable"
    );
    // ...and discoverable via search_tools (empty embedding -> FTS fallback).
    let results = store
        .search_tools("telemetry", vec![], 10)
        .await
        .expect("fts search after enable");
    assert!(
        results.iter().any(|t| t.name == "serverb__frobnicate"),
        "search_tools must surface the newly-enabled tool; got {:?}",
        results.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn reindex_source_removes_disabled_server_tools() {
    let Some(fx) = fixture("reindex_remove").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; reindex_source_removes_disabled_server_tools");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // Both servers enabled.
    let mut superset = server_a_tools();
    superset.push(server_b_tool());
    reindex_mcp(&store, superset).await;
    assert!(
        store
            .tool_definition("serverb__frobnicate")
            .await
            .expect("lookup before disable")
            .is_some(),
        "precondition: serverb tool present before disable"
    );

    // Disable server B: reindex with the reduced set {A}.
    reindex_mcp(&store, server_a_tools()).await;

    // Server B's rows are pruned...
    assert!(
        store
            .tool_definition("serverb__frobnicate")
            .await
            .expect("lookup after disable")
            .is_none(),
        "disabled server's row must be removed from the index"
    );
    // ...and no longer surfaced by search...
    let results = store
        .search_tools("telemetry", vec![], 10)
        .await
        .expect("fts search after disable");
    assert!(
        !results.iter().any(|t| t.name == "serverb__frobnicate"),
        "disabled tool must not be searchable"
    );
    // ...while server A's rows remain.
    assert!(
        store
            .tool_definition("servera__alpha")
            .await
            .expect("lookup surviving tool")
            .is_some(),
        "the still-enabled server's rows must survive the reindex"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn reindex_source_is_idempotent() {
    let Some(fx) = fixture("reindex_idem").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; reindex_source_is_idempotent");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    let full_set = || {
        let mut tools = server_a_tools();
        tools.push(server_b_tool());
        tools
    };

    reindex_mcp(&store, full_set()).await;
    let first = mcp_source_names(&fx).await;

    // Repeating the same reindex must converge on the identical row set.
    reindex_mcp(&store, full_set()).await;
    let second = mcp_source_names(&fx).await;

    assert_eq!(
        first,
        vec![
            "servera__alpha".to_string(),
            "servera__beta".to_string(),
            "serverb__frobnicate".to_string(),
        ],
        "first reindex must register exactly the full set"
    );
    assert_eq!(
        first, second,
        "a repeated reindex must yield the same row set (idempotent)"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn reindex_source_empty_set_clears_all_mcp_rows() {
    let Some(fx) = fixture("reindex_empty").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; reindex_source_empty_set_clears_all_mcp_rows");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // Populate: both servers enabled.
    let mut superset = server_a_tools();
    superset.push(server_b_tool());
    reindex_mcp(&store, superset).await;
    assert!(
        !mcp_source_names(&fx).await.is_empty(),
        "precondition: mcp rows present before the empty reindex"
    );

    // Disable the last MCP server (or a zero-tool server): reindex with an empty
    // set -> the delete runs, the INSERT batch is empty, no mcp rows survive.
    reindex_mcp(&store, vec![]).await;

    assert!(
        mcp_source_names(&fx).await.is_empty(),
        "an empty reindex must clear every mcp row"
    );
    // Nothing is left to surface: the FTS token that singled out server B before
    // now matches no row.
    let results = store
        .search_tools("telemetry", vec![], 10)
        .await
        .expect("fts search after empty reindex");
    assert!(
        results.is_empty(),
        "no tool may remain searchable after an empty reindex; got {:?}",
        results.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    fx.cleanup().await;
}
