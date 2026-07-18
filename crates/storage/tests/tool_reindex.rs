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
//! - `reindex_source_rolls_back_on_mid_batch_failure` — the atomic
//!   `reindex_source` (#519): when a provider batch *after* the first fails
//!   mid-loop, the whole reindex rolls back to the prior good state — no provider
//!   swept-but-not-restored (dropped), and none of the partial new state leaked
//!   in.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`).

mod support;

use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
use desktop_assistant_storage::{PgToolRegistryStore, ToolRegisterBatch};
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
        .register_tools(tools, "mcp", false, None, embeddings, None)
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

/// One provider batch with NULL embeddings — the same mapping the daemon's
/// hot-reindex closure feeds to `reindex_source`.
fn batch(provider: &str, tools: Vec<ToolDefinition>) -> ToolRegisterBatch {
    let embeddings = vec![None; tools.len()];
    ToolRegisterBatch {
        tools,
        is_core: false,
        provider: Some(provider.to_string()),
        embeddings,
        embedding_model: None,
    }
}

/// A tool whose description carries an embedded NUL byte. Postgres rejects NUL
/// in `text` (`invalid byte sequence for encoding "UTF8": 0x00`), so INSERTing
/// this row fails *inside* the reindex transaction — a faithful stand-in for a
/// transient mid-loop DB error striking the Nth provider, after the sweep and
/// the earlier batches have already been applied within the transaction.
fn poison_tool() -> ToolDefinition {
    ToolDefinition::new(
        "serverpoison__boom",
        "bad\u{0}description",
        serde_json::json!({"type": "object"}),
    )
}

#[tokio::test]
async fn reindex_source_rolls_back_on_mid_batch_failure() {
    let Some(fx) = fixture("reindex_rollback").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; reindex_source_rolls_back_on_mid_batch_failure"
        );
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // Prior good state: two providers, registered atomically.
    store
        .reindex_source(
            "mcp",
            vec![
                batch("mcp:servera", server_a_tools()),
                batch("mcp:serverb", vec![server_b_tool()]),
            ],
        )
        .await
        .expect("seed prior good state");
    let before = mcp_source_names(&fx).await;
    assert_eq!(
        before,
        vec![
            "servera__alpha".to_string(),
            "servera__beta".to_string(),
            "serverb__frobnicate".to_string(),
        ],
        "precondition: the prior good state has all three rows"
    );

    // Attempt a NEW reindex in which a provider *after* the first fails mid-loop.
    // Ordering matters: a good new provider (serverc) is applied first, then the
    // poison provider fails. With a non-atomic loop the sweep + the serverc
    // insert would already be committed, so serverc would survive while serverb
    // (swept, never re-registered) would vanish — a partial index. The atomic
    // `reindex_source` must instead roll the whole thing back.
    let serverc = ToolDefinition::new(
        "serverc__gamma",
        "A newly enabled provider tool",
        serde_json::json!({"type": "object"}),
    );
    let err = store
        .reindex_source(
            "mcp",
            vec![
                batch("mcp:serverc", vec![serverc]),
                batch("mcp:serverpoison", vec![poison_tool()]),
            ],
        )
        .await
        .expect_err("a mid-loop INSERT failure must surface as an error, not be swallowed");
    // The failure is surfaced to the caller (the mcp-client boundary swallows it,
    // not the store) — assert it is a storage error, not a silent Ok.
    assert!(
        matches!(err, desktop_assistant_core::CoreError::Storage(_)),
        "a mid-loop DB failure surfaces as CoreError::Storage; got {err:?}"
    );

    // Invariant (atomic / all-or-nothing): the index is byte-for-byte the prior
    // good state. No provider was silently dropped, and none of the partial new
    // state leaked in.
    let after = mcp_source_names(&fx).await;
    assert_eq!(
        after, before,
        "a mid-loop failure must roll the reindex back to the prior good state"
    );
    assert!(
        store
            .tool_definition("serverc__gamma")
            .await
            .expect("lookup serverc after rollback")
            .is_none(),
        "the partially-applied new provider must NOT survive a rolled-back reindex"
    );
    assert!(
        store
            .tool_definition("serverb__frobnicate")
            .await
            .expect("lookup serverb after rollback")
            .is_some(),
        "a provider present before the failed reindex must NOT be dropped by it"
    );

    fx.cleanup().await;
}
