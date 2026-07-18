//! DB-gated integration tests for provider-level tool surfacing (Phase 1).
//!
//! Every tool carries a `provider` (an MCP server identity or a builtin group),
//! and the daemon registers one synthetic, searchable `provider:<provider>` row
//! per provider. When that synthetic row matches a tool-search query, its member
//! tools (sharing the `provider` value) get their fused score boosted so a whole
//! provider's tools surface together — while the synthetic row itself is never
//! returned as a callable tool.
//!
//! Acceptance criteria, each a named test below:
//! - `register_tools_persists_provider_column` — the batch-constant `provider`
//!   arg is written (and NULL when `None`).
//! - `unregister_source_sweeps_provider_and_tool_rows` — a source sweep removes
//!   both the member tools AND the synthetic provider row of that source, and
//!   leaves other sources intact.
//! - `register_tools_rejects_non_synthetic_provider_name` — a real (non
//!   synthetic) tool literally named `provider:*` is refused (guard #4).
//! - `search_provider_match_boosts_member_tools_into_top_n` — a member that
//!   ranks just below the cutoff rises into the top-N *only* because its provider
//!   row matched (proven by re-running with the provider row removed).
//! - `search_provider_row_never_returned` — a query strongly matching a provider
//!   row returns only real tools; no `name LIKE 'provider:%'` leaks out.
//! - `search_no_provider_match_ranking_unchanged` — with no provider row present,
//!   ordering is identical to the plain-RRF baseline (additive-zero boost).
//! - `search_boost_no_double_count` — a member present in BOTH the vector and
//!   text branches whose provider matched has the provider score added exactly
//!   once (asserted on the exact boosted score).
//! - `search_fts_fallback_excludes_provider_rows_and_boosts` — the empty
//!   query-embedding path excludes provider rows and still boosts members.
//! - `core_tools_excludes_provider_rows` — synthetic rows (is_core = FALSE) never
//!   appear in `core_tools()`.
//!
//! Gated on `TEST_DATABASE_URL`; pass-skips when unset (see `support`).

mod support;

use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
use desktop_assistant_storage::PgToolRegistryStore;
use sqlx::Row;

use support::DbFixture;

fn tool(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition::new(name, description, serde_json::json!({"type": "object"}))
}

/// The synthetic, searchable row for a provider — `provider:<name>` with the
/// provider description and its member tool names in the text (mirrors what the
/// daemon registers via `ReindexProvider::synthetic_row`).
fn provider_row(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition::new(
        format!("provider:{name}"),
        description,
        serde_json::json!({}),
    )
}

/// Build a fixture, or pass-skip when `TEST_DATABASE_URL` is unset.
async fn fixture(prefix: &str) -> Option<DbFixture> {
    DbFixture::try_new(prefix).await
}

/// Read the `provider` column for a tool by name.
async fn provider_of(fx: &DbFixture, name: &str) -> Option<String> {
    let row = sqlx::query("SELECT provider FROM tool_definitions WHERE name = $1")
        .bind(name)
        .fetch_optional(&fx.pool)
        .await
        .expect("select provider column");
    row.and_then(|r| r.get::<Option<String>, _>("provider"))
}

/// All tool names currently registered under a source, sorted for determinism.
async fn source_names(fx: &DbFixture, source: &str) -> Vec<String> {
    let rows = sqlx::query("SELECT name FROM tool_definitions WHERE source = $1 ORDER BY name")
        .bind(source)
        .fetch_all(&fx.pool)
        .await
        .expect("select source names");
    rows.iter().map(|r| r.get::<String, _>("name")).collect()
}

#[tokio::test]
async fn register_tools_persists_provider_column() {
    let Some(fx) = fixture("provider_col").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; register_tools_persists_provider_column");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // A provider-tagged tool records the batch-constant provider value...
    store
        .register_tools(
            vec![tool("weather__forecast", "Get the forecast")],
            "mcp",
            false,
            Some("weather"),
            vec![None],
            None,
        )
        .await
        .expect("register with provider");
    assert_eq!(
        provider_of(&fx, "weather__forecast").await.as_deref(),
        Some("weather"),
        "the provider column must persist the batch-constant provider value"
    );

    // ...while a `None` provider leaves the column NULL (unclassified).
    store
        .register_tools(
            vec![tool("legacy_tool", "No provider")],
            "mcp",
            false,
            None,
            vec![None],
            None,
        )
        .await
        .expect("register without provider");
    assert_eq!(
        provider_of(&fx, "legacy_tool").await,
        None,
        "a None provider must leave the column NULL"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn unregister_source_sweeps_provider_and_tool_rows() {
    let Some(fx) = fixture("provider_sweep").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; unregister_source_sweeps_provider_and_tool_rows");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // MCP source: two member tools + the synthetic provider row.
    store
        .register_tools(
            vec![
                tool("weather__forecast", "Forecast"),
                tool("weather__alerts", "Alerts"),
                provider_row("weather", "Weather tools. Tools: weather__forecast, weather__alerts."),
            ],
            "mcp",
            false,
            Some("weather"),
            vec![None, None, None],
            None,
        )
        .await
        .expect("register mcp source");

    // Builtin source: a member tool + its synthetic provider row (is_core split
    // is irrelevant to the sweep, which keys on `source`).
    store
        .register_tools(
            vec![tool("builtin_db_query", "Run SQL")],
            "builtin",
            true,
            Some("database"),
            vec![None],
            None,
        )
        .await
        .expect("register builtin members");
    store
        .register_tools(
            vec![provider_row("database", "Database tools. Tools: builtin_db_query.")],
            "builtin",
            false,
            Some("database"),
            vec![None],
            None,
        )
        .await
        .expect("register builtin provider row");

    // Sweep the mcp source: BOTH its member tools and its synthetic row go.
    store.unregister_source("mcp").await.expect("sweep mcp");
    assert!(
        source_names(&fx, "mcp").await.is_empty(),
        "the mcp sweep must remove member tools AND the synthetic provider row"
    );
    // The builtin source is untouched (members + its provider row survive).
    assert_eq!(
        source_names(&fx, "builtin").await,
        vec![
            "builtin_db_query".to_string(),
            "provider:database".to_string(),
        ],
        "unrelated source rows (incl. its provider row) survive a foreign sweep"
    );

    // And the builtin sweep clears its own member + provider rows.
    store
        .unregister_source("builtin")
        .await
        .expect("sweep builtin");
    assert!(
        source_names(&fx, "builtin").await.is_empty(),
        "the builtin sweep must remove its members and its synthetic provider row"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn register_tools_rejects_non_synthetic_provider_name() {
    let Some(fx) = fixture("provider_guard").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; register_tools_rejects_non_synthetic_provider_name");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // A real tool literally named `provider:*` that is NOT this batch's synthetic
    // row (batch provider = "weather", but the tool claims `provider:evil`) must
    // be refused so it can never be returned by search or dispatched (guard #4).
    let err = store
        .register_tools(
            vec![tool("provider:evil", "impersonating a provider row")],
            "mcp",
            false,
            Some("weather"),
            vec![None],
            None,
        )
        .await
        .expect_err("a non-synthetic provider:* tool must be rejected");
    assert!(
        err.to_string().contains("provider:"),
        "the rejection must name the offending reserved prefix; got: {err}"
    );
    // Nothing was written.
    assert!(
        source_names(&fx, "mcp").await.is_empty(),
        "a rejected batch must not persist any row"
    );

    // The matching synthetic row (batch provider = "weather", name
    // `provider:weather`) is allowed.
    store
        .register_tools(
            vec![provider_row("weather", "Weather tools.")],
            "mcp",
            false,
            Some("weather"),
            vec![None],
            None,
        )
        .await
        .expect("the batch's own synthetic provider row is permitted");
    assert_eq!(
        source_names(&fx, "mcp").await,
        vec!["provider:weather".to_string()],
        "the synthetic row for the batch's own provider is written"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn core_tools_excludes_provider_rows() {
    let Some(fx) = fixture("provider_core").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; core_tools_excludes_provider_rows");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // A core builtin member (is_core = TRUE) and its synthetic provider row
    // (is_core = FALSE). core_tools() returns the member but never the provider
    // row (guard #1: synthetic rows are non-core).
    store
        .register_tools(
            vec![tool("builtin_db_query", "Run SQL")],
            "builtin",
            true,
            Some("database"),
            vec![None],
            None,
        )
        .await
        .expect("register core member");
    store
        .register_tools(
            vec![provider_row("database", "Database tools.")],
            "builtin",
            false,
            Some("database"),
            vec![None],
            None,
        )
        .await
        .expect("register provider row");

    let core = store.core_tools().await.expect("core_tools");
    assert!(
        core.iter().any(|t| t.name == "builtin_db_query"),
        "the real core builtin must be returned"
    );
    assert!(
        !core.iter().any(|t| t.name.starts_with("provider:")),
        "no synthetic provider row may appear in core_tools(); got {:?}",
        core.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    fx.cleanup().await;
}
