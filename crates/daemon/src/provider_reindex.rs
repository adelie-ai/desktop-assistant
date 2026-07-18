//! Provider-grouped registration for the persistent tool-search index.
//!
//! Turns the reindex boundary's [`ReindexProvider`] groups (MCP servers) and the
//! builtin groups (at startup) into `register_tools` batches: each provider's
//! member tools plus its synthetic `provider:<name>` row, so a tool-search hit on
//! the provider row boosts its members. Kept as pure builders plus a thin apply
//! so the batch shapes are unit-testable without a database.

use std::collections::BTreeMap;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::{ReindexProvider, ToolRegistryStore};
use desktop_assistant_mcp_client::executor::BuiltinToolService;

/// Generic provider group + blurb for any builtin the map somehow missed, so a
/// new builtin is never dropped from registration. `provider_group` classifies
/// every real builtin (guarded by `builtin_provider_map_is_exhaustive`), so this
/// is only reached if a new builtin ships without a mapping.
const BUILTIN_FALLBACK_GROUP: &str = "builtin";
const BUILTIN_FALLBACK_BLURB: &str = "Adele's built-in tools.";

/// One `register_tools` call's worth of rows: a batch sharing a source, an
/// `is_core` flag, and a provider identity.
pub(crate) struct RegisterBatch {
    pub tools: Vec<ToolDefinition>,
    pub source: &'static str,
    pub is_core: bool,
    pub provider: String,
}

/// Build the registration batches for MCP providers: for each provider, its
/// member tools plus its synthetic `provider:<name>` row, all non-core under the
/// provider's source (`"mcp"`). Exactly one synthetic row per provider group.
pub(crate) fn build_mcp_batches(providers: Vec<ReindexProvider>) -> Vec<RegisterBatch> {
    providers
        .into_iter()
        .map(|p| {
            let synthetic = p.synthetic_row();
            let provider = p.qualified_provider();
            let mut tools = p.tools;
            tools.push(synthetic);
            RegisterBatch {
                tools,
                source: p.source,
                is_core: false,
                provider,
            }
        })
        .collect()
}

/// Build the registration batches for the builtin tools, grouped by provider.
/// For each group: one batch of member tools registered **core** (`is_core =
/// true`, always sent to the LLM) plus one batch holding only the synthetic
/// `provider:<group>` row registered **non-core** (`is_core = false`, so it is
/// searchable and drives the boost but is never sent as a callable tool). This
/// surfaces builtins to tool-search by the SAME provider mechanism as MCP servers.
///
/// A builtin the classifier doesn't recognize still gets registered under a
/// generic fallback group (never dropped); the exhaustiveness test keeps every
/// real builtin off that path.
pub(crate) fn build_builtin_batches(defs: Vec<ToolDefinition>) -> Vec<RegisterBatch> {
    // Group members by provider. BTreeMap for deterministic batch ordering.
    let mut groups: BTreeMap<&'static str, Vec<ToolDefinition>> = BTreeMap::new();
    for def in defs {
        let provider =
            BuiltinToolService::provider_group(&def.name).unwrap_or(BUILTIN_FALLBACK_GROUP);
        groups.entry(provider).or_default().push(def);
    }

    let mut batches = Vec::new();
    for (provider, members) in groups {
        let blurb = BuiltinToolService::provider_blurb(provider).unwrap_or(BUILTIN_FALLBACK_BLURB);
        let group = ReindexProvider {
            name: provider.to_string(),
            source: "builtin",
            description: blurb.to_string(),
            tools: members,
        };
        let synthetic = group.synthetic_row();
        let qualified = group.qualified_provider();
        // Members are core; the synthetic provider row is non-core.
        batches.push(RegisterBatch {
            tools: group.tools,
            source: "builtin",
            is_core: true,
            provider: qualified.clone(),
        });
        batches.push(RegisterBatch {
            tools: vec![synthetic],
            source: "builtin",
            is_core: false,
            provider: qualified,
        });
    }
    batches
}

/// Register a set of batches with NULL embeddings (the background backfill fills
/// vectors later). Batches are applied in order; the first error stops the run.
pub(crate) async fn apply_batches<S: ToolRegistryStore>(
    store: &S,
    batches: Vec<RegisterBatch>,
) -> Result<(), CoreError> {
    for batch in batches {
        let embeddings = vec![None; batch.tools.len()];
        store
            .register_tools(
                batch.tools,
                batch.source,
                batch.is_core,
                Some(&batch.provider),
                embeddings,
                None,
            )
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(name: &str) -> ToolDefinition {
        ToolDefinition::new(name, format!("{name} does things"), serde_json::json!({}))
    }

    #[test]
    fn reindex_emits_one_provider_row_per_mcp_provider() {
        let providers = vec![
            ReindexProvider {
                name: "weather".into(),
                source: "mcp",
                description: "Weather and forecasts.".into(),
                tools: vec![td("weather__forecast"), td("weather__alerts")],
            },
            ReindexProvider {
                name: "geocode".into(),
                source: "mcp",
                description: "Place lookup.".into(),
                tools: vec![td("geocode__search")],
            },
        ];
        let batches = build_mcp_batches(providers);
        assert_eq!(batches.len(), 2, "one batch per provider group");

        for b in &batches {
            assert_eq!(b.source, "mcp", "MCP batches use the mcp source");
            assert!(!b.is_core, "MCP provider batches are non-core");
            let rows: Vec<&str> = b
                .tools
                .iter()
                .filter(|t| t.name.starts_with("provider:"))
                .map(|t| t.name.as_str())
                .collect();
            assert_eq!(
                rows,
                vec![format!("provider:{}", b.provider).as_str()],
                "exactly one synthetic provider row per group, named provider:<name>"
            );
        }

        let synthetic_total = batches
            .iter()
            .flat_map(|b| b.tools.iter())
            .filter(|t| t.name.starts_with("provider:"))
            .count();
        assert_eq!(
            synthetic_total, 2,
            "the reindex emits exactly one provider row per MCP provider"
        );

        let weather = batches
            .iter()
            .find(|b| b.provider == "mcp:weather")
            .expect("weather batch (source-qualified provider)");
        assert_eq!(
            weather.tools.len(),
            3,
            "the group carries its 2 member tools plus 1 synthetic row"
        );
    }

    #[test]
    fn startup_registers_builtins_grouped_with_provider_rows() {
        // Drive the real default builtin set through the grouping.
        let defs = BuiltinToolService::new().tool_definitions();
        assert!(!defs.is_empty(), "the default builtin set is non-empty");
        let batches = build_builtin_batches(defs);

        // Every batch is under the builtin source; every member is classified
        // (no batch lands in the generic fallback group for the default set).
        // Provider identities are source-qualified (`builtin:<group>`).
        let expected_groups: Vec<String> = BuiltinToolService::PROVIDER_GROUPS
            .iter()
            .map(|(id, _)| format!("builtin:{id}"))
            .collect();
        for b in &batches {
            assert_eq!(b.source, "builtin");
            assert!(
                expected_groups.contains(&b.provider),
                "unexpected builtin group '{}' (fell back?)",
                b.provider
            );
        }

        // Member batches are core; synthetic batches are a single non-core
        // provider:<group> row. Exactly one synthetic row per group.
        for b in &batches {
            let is_synthetic = b.tools.iter().all(|t| t.name.starts_with("provider:"));
            if is_synthetic {
                assert!(!b.is_core, "the synthetic provider row must be non-core");
                assert_eq!(b.tools.len(), 1, "synthetic batch holds exactly one row");
                assert_eq!(b.tools[0].name, format!("provider:{}", b.provider));
            } else {
                assert!(b.is_core, "builtin member tools are core");
                assert!(
                    b.tools.iter().all(|t| t.name.starts_with("builtin_")),
                    "member batch holds only real builtins"
                );
            }
        }

        // One synthetic provider row per group present, and every default group
        // (knowledge/scratchpad/database/recall/system/tool-meta) is represented,
        // each source-qualified as builtin:<group>.
        let synthetic_providers: std::collections::BTreeSet<String> = batches
            .iter()
            .flat_map(|b| b.tools.iter())
            .filter(|t| t.name.starts_with("provider:"))
            .map(|t| t.name.strip_prefix("provider:").unwrap().to_string())
            .collect();
        assert_eq!(
            synthetic_providers,
            expected_groups.into_iter().collect(),
            "every builtin group must get exactly one synthetic provider row"
        );
    }

    #[test]
    fn mcp_and_builtin_same_bare_name_get_distinct_providers() {
        // FIX 3 (keyspace collision): an MCP server literally named "database"
        // must not collide with the builtin "database" group. Their provider
        // identities and synthetic row names must be source-qualified and distinct.
        let mcp = build_mcp_batches(vec![ReindexProvider {
            name: "database".into(),
            source: "mcp",
            description: "External DB server.".into(),
            tools: vec![td("database__query")],
        }]);
        let builtin = build_builtin_batches(vec![ToolDefinition::new(
            "builtin_db_query",
            "Run SQL",
            serde_json::json!({}),
        )]);

        let synthetic = |batches: &[RegisterBatch]| -> Vec<String> {
            batches
                .iter()
                .flat_map(|b| b.tools.iter())
                .filter(|t| t.name.starts_with("provider:"))
                .map(|t| t.name.clone())
                .collect()
        };
        let mcp_rows = synthetic(&mcp);
        let builtin_rows = synthetic(&builtin);

        assert!(
            mcp_rows.contains(&"provider:mcp:database".to_string()),
            "mcp server 'database' -> provider:mcp:database; got {mcp_rows:?}"
        );
        assert!(
            builtin_rows.contains(&"provider:builtin:database".to_string()),
            "builtin group 'database' -> provider:builtin:database; got {builtin_rows:?}"
        );
        // The provider *column* values are distinct too, so the boost join never
        // cross-matches an mcp member to a builtin provider row or vice versa.
        assert_eq!(mcp[0].provider, "mcp:database");
        for b in &builtin {
            assert_ne!(
                b.provider, "mcp:database",
                "no builtin batch may share the mcp provider identity"
            );
        }
        // No synthetic row name is shared between the two sources.
        for r in &mcp_rows {
            assert!(
                !builtin_rows.contains(r),
                "row '{r}' must not appear under both sources"
            );
        }
    }
}
