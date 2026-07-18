//! Provider-grouped registration for the persistent tool-search index.
//!
//! Turns the reindex boundary's [`ReindexProvider`] groups (MCP servers) and the
//! builtin groups (at startup) into `register_tools` batches: each provider's
//! member tools plus its synthetic `provider:<name>` row, so a tool-search hit on
//! the provider row boosts its members. Kept as pure builders plus a thin apply
//! so the batch shapes are unit-testable without a database.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::{ReindexProvider, ToolRegistryStore};

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
            let mut tools = p.tools;
            tools.push(synthetic);
            RegisterBatch {
                tools,
                source: p.source,
                is_core: false,
                provider: p.name,
            }
        })
        .collect()
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
            .find(|b| b.provider == "weather")
            .expect("weather batch");
        assert_eq!(
            weather.tools.len(),
            3,
            "the group carries its 2 member tools plus 1 synthetic row"
        );
    }
}
