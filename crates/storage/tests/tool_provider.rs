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
        eprintln!(
            "skip: TEST_DATABASE_URL not set; unregister_source_sweeps_provider_and_tool_rows"
        );
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // MCP source: two member tools + the synthetic provider row.
    store
        .register_tools(
            vec![
                tool("weather__forecast", "Forecast"),
                tool("weather__alerts", "Alerts"),
                provider_row(
                    "weather",
                    "Weather tools. Tools: weather__forecast, weather__alerts.",
                ),
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
            vec![provider_row(
                "database",
                "Database tools. Tools: builtin_db_query.",
            )],
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
        eprintln!(
            "skip: TEST_DATABASE_URL not set; register_tools_rejects_non_synthetic_provider_name"
        );
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

/// One 2-D embedding chunk, wrapped for the `Vec<Option<Vec<Vec<f32>>>>` shape.
fn embed(x: f32, y: f32) -> Option<Vec<Vec<f32>>> {
    Some(vec![vec![x, y]])
}

/// Delete a single row by name (used to remove a provider row for the
/// counterfactual "would-not-without-the-boost" half of a test).
async fn delete_row(fx: &DbFixture, name: &str) {
    sqlx::query("DELETE FROM tool_definitions WHERE name = $1")
        .bind(name)
        .execute(&fx.pool)
        .await
        .expect("delete row");
}

#[tokio::test]
async fn search_provider_match_boosts_member_tools_into_top_n() {
    let Some(fx) = fixture("provider_boost_topn").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; search_provider_match_boosts_member_tools_into_top_n"
        );
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // Ten filler tools whose embeddings sit progressively farther from the query
    // vector [1,0] (cosine distance grows with the y-tilt), so their vector ranks
    // are a stable 1..10. None match the text query. Provider = None (unboosted).
    let fillers: Vec<ToolDefinition> = (1..=10)
        .map(|k| tool(&format!("filler__{k:02}"), "generic filler capability"))
        .collect();
    let filler_embeddings: Vec<Option<Vec<Vec<f32>>>> =
        (1..=10).map(|k| embed(1.0, 0.01 * k as f32)).collect();
    store
        .register_tools(fillers, "mcp", false, None, filler_embeddings, None)
        .await
        .expect("register fillers");

    // The target member: vector rank 11 (just past filler_10 at y=0.10), no text
    // match — so on its own it sits ONE below a top-10 cutoff. Its provider = boostme.
    store
        .register_tools(
            vec![tool("boostme__member", "an unrelated gizmo")],
            "mcp",
            false,
            Some("boostme"),
            vec![embed(1.0, 0.11)],
            None,
        )
        .await
        .expect("register member");

    // The provider row matches the text query strongly (text-only; NULL embedding).
    store
        .register_tools(
            vec![provider_row(
                "boostme",
                "Frobnication service. Tools: boostme__member.",
            )],
            "mcp",
            false,
            Some("boostme"),
            vec![None],
            None,
        )
        .await
        .expect("register provider row");

    // Query: text hits ONLY the provider row ("frobnication"); embedding [1,0].
    let names_with_boost: Vec<String> = store
        .search_tools("frobnication", vec![1.0, 0.0], 10)
        .await
        .expect("search with boost")
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        names_with_boost.contains(&"boostme__member".to_string()),
        "the provider match must lift its member into the top-10; got {names_with_boost:?}"
    );
    assert!(
        !names_with_boost.iter().any(|n| n.starts_with("provider:")),
        "the provider row itself must never be returned; got {names_with_boost:?}"
    );

    // Counterfactual: remove the provider row and re-run the identical query. With
    // no provider match the member falls back to its raw rank-11 and drops out of
    // the top-10 — proving the boost (not the raw rank) put it there.
    delete_row(&fx, "provider:boostme").await;
    let names_without_boost: Vec<String> = store
        .search_tools("frobnication", vec![1.0, 0.0], 10)
        .await
        .expect("search without boost")
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        !names_without_boost.contains(&"boostme__member".to_string()),
        "without the provider row the member must NOT be in the top-10; got {names_without_boost:?}"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn search_provider_row_never_returned() {
    let Some(fx) = fixture("provider_never_returned").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; search_provider_row_never_returned");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    store
        .register_tools(
            vec![tool("svc__do_thing", "handle a special widget request")],
            "mcp",
            false,
            Some("svc"),
            vec![embed(1.0, 0.0)],
            None,
        )
        .await
        .expect("register member");
    // A provider row whose text matches the query even harder than the member.
    store
        .register_tools(
            vec![provider_row(
                "svc",
                "Special widget service tools. Tools: svc__do_thing.",
            )],
            "mcp",
            false,
            Some("svc"),
            vec![None],
            None,
        )
        .await
        .expect("register provider row");

    // Query strongly matches the provider row (and the member).
    let results = store
        .search_tools("special widget service tools", vec![1.0, 0.0], 10)
        .await
        .expect("search");
    assert!(
        results.iter().any(|t| t.name == "svc__do_thing"),
        "the real member tool must be returned; got {:?}",
        results.iter().map(|t| &t.name).collect::<Vec<_>>()
    );
    assert!(
        !results.iter().any(|t| t.name.starts_with("provider:")),
        "no `provider:*` row may ever be returned as a tool; got {:?}",
        results.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn search_no_provider_match_ranking_unchanged() {
    let Some(fx) = fixture("provider_no_match").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; search_no_provider_match_ranking_unchanged");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // No provider rows at all -> matched_providers is empty -> the boost is
    // additive-zero -> ordering is the plain-RRF baseline. Construct a case that
    // genuinely exercises vector+text fusion with distinct, integer-rank scores:
    //   toolB: vector rank 2 + text rank 1  -> 1/62 + 1/61  (highest)
    //   toolA: vector rank 1, no text       -> 1/61
    //   toolC: no vector, text rank 2        -> 1/62         (lowest)
    // Expected order: [B, A, C].
    store
        .register_tools(
            vec![tool("tool_a", "beta gamma capability")],
            "mcp",
            false,
            None,
            vec![embed(1.0, 0.0)], // closest to query -> vector rank 1
            None,
        )
        .await
        .expect("register A");
    store
        .register_tools(
            vec![tool("tool_b", "alpha alpha alpha alpha task")],
            "mcp",
            false,
            None,
            vec![embed(0.9, 0.1)], // second closest -> vector rank 2
            None,
        )
        .await
        .expect("register B");
    store
        .register_tools(
            vec![tool("tool_c", "alpha task")], // fewer 'alpha' -> weaker text rank
            "mcp",
            false,
            None,
            vec![None], // no embedding -> text branch only
            None,
        )
        .await
        .expect("register C");

    let order: Vec<String> = store
        .search_tools("alpha", vec![1.0, 0.0], 10)
        .await
        .expect("search")
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(
        order,
        vec![
            "tool_b".to_string(),
            "tool_a".to_string(),
            "tool_c".to_string()
        ],
        "with no provider row the boost is additive-zero, so ordering must equal \
         the plain-RRF baseline [B, A, C]"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn search_boost_no_double_count() {
    let Some(fx) = fixture("provider_no_double").await else {
        eprintln!("skip: TEST_DATABASE_URL not set; search_boost_no_double_count");
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // The member matches BOTH branches: vector rank 1 (it IS the query vector) and
    // text rank 2 (the provider row, with more term repetition, takes text rank 1).
    //   member raw fused = 1/61 (vector) + 1/62 (text)
    //   provider_score   = 1/61 (provider row text rank 1)
    //   correct boosted  = (1/61 + 1/62) + 1/61   [provider score added ONCE]
    //   double-count bug = (1/61 + 1/62) + 2/61
    store
        .register_tools(
            vec![tool("dup__tool", "special dup gadget")],
            "mcp",
            false,
            Some("dup"),
            vec![embed(1.0, 0.0)],
            None,
        )
        .await
        .expect("register member");
    store
        .register_tools(
            vec![provider_row(
                "dup",
                "special special special dup dup dup gadget gadget gadget tools",
            )],
            "mcp",
            false,
            Some("dup"),
            vec![None],
            None,
        )
        .await
        .expect("register provider row");

    let scored = store
        .search_tools_scored("special dup gadget", vec![1.0, 0.0], 10)
        .await
        .expect("scored search");
    let member = scored
        .iter()
        .find(|(t, _)| t.name == "dup__tool")
        .map(|(_, s)| *s)
        .expect("member present in results");

    let one = 1.0_f64 / 61.0;
    let two = 1.0_f64 / 62.0;
    let expected = (one + two) + one; // provider score added exactly once
    let double = (one + two) + 2.0 * one;
    assert!(
        (member - expected).abs() < 1e-9,
        "provider score must be added exactly once: got {member}, expected {expected} \
         (double-count would be {double})"
    );

    fx.cleanup().await;
}

#[tokio::test]
async fn search_fts_fallback_excludes_provider_rows_and_boosts() {
    let Some(fx) = fixture("provider_fts_fallback").await else {
        eprintln!(
            "skip: TEST_DATABASE_URL not set; search_fts_fallback_excludes_provider_rows_and_boosts"
        );
        return;
    };
    let store = PgToolRegistryStore::new(fx.pool.clone());

    // Empty query embedding -> the FTS-only fallback path. `other` out-ranks the
    // `member` on raw ts_rank (more term repetition), but the member's provider
    // row matches strongest of all, so the boost lifts the member above `other`.
    store
        .register_tools(
            vec![tool("fb__member", "widget")], // 1 occurrence -> weakest raw rank
            "mcp",
            false,
            Some("fb"),
            vec![None],
            None,
        )
        .await
        .expect("register member");
    store
        .register_tools(
            vec![tool("other__tool", "widget widget widget widget")], // stronger raw
            "mcp",
            false,
            None,
            vec![None],
            None,
        )
        .await
        .expect("register other");
    store
        .register_tools(
            vec![provider_row(
                "fb",
                "widget widget widget widget widget widget widget widget widget widget widget widget",
            )],
            "mcp",
            false,
            Some("fb"),
            vec![None],
            None,
        )
        .await
        .expect("register provider row");

    // Empty embedding forces the FTS fallback.
    let names: Vec<String> = store
        .search_tools("widget", vec![], 10)
        .await
        .expect("fts fallback search")
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        !names.iter().any(|n| n.starts_with("provider:")),
        "the FTS fallback MUST exclude provider rows; got {names:?}"
    );
    let pos = |n: &str| names.iter().position(|x| x == n);
    assert!(
        pos("fb__member") < pos("other__tool"),
        "the provider boost must lift the member above the higher-raw-rank tool \
         in the FTS fallback; got {names:?}"
    );

    // Counterfactual: drop the provider row; the raw ts_rank order reasserts
    // itself (other above member), proving the reordering came from the boost.
    delete_row(&fx, "provider:fb").await;
    let baseline: Vec<String> = store
        .search_tools("widget", vec![], 10)
        .await
        .expect("fts fallback baseline")
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        pos_in(&baseline, "other__tool") < pos_in(&baseline, "fb__member"),
        "without the provider row the raw ts_rank order must put other above member; \
         got {baseline:?}"
    );

    fx.cleanup().await;
}

/// Position of `name` in `names`, or `usize::MAX` when absent (so a missing name
/// sorts last in the counterfactual comparison).
fn pos_in(names: &[String], name: &str) -> usize {
    names.iter().position(|x| x == name).unwrap_or(usize::MAX)
}
