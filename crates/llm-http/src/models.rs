//! Shared curated-model / live-endpoint merge resolver.
//!
//! Issue #304. Before this, the connectors with a hand-maintained curated
//! model table (`llm-anthropic`, `llm-openai`) each carried the same
//! `TODO(#7)` about merging the static curated list with the provider's live
//! `/v1/models` endpoint, and would have hand-rolled the merge independently —
//! letting the "curated wins, unknown live appended" policy drift per provider.
//!
//! [`merge_curated_with_live`] is the single place that policy lives:
//!
//! * the curated entries are emitted first, in their original order, and their
//!   metadata (display name, context window, capability flags) wins on any id
//!   overlap with the live list — the live `/v1/models` response typically
//!   carries only bare ids with no capability metadata, so the curated table
//!   remains the source of truth for models we know about;
//! * live models whose id is not in the curated table are appended afterwards,
//!   in the order the endpoint returned them, so freshly released models still
//!   surface in the picker without a code change.
//!
//! Connectors that fetch a live list pass it in; connectors that don't yet hit
//! their list endpoint pass an empty `live` slice and get their curated table
//! back unchanged, ready to grow a live fetch later without re-deriving the
//! merge policy.

use desktop_assistant_core::ports::llm::ModelInfo;

/// Merge a curated model table with a live endpoint listing.
///
/// Curated entries come first (preserving their order) and win on any id
/// overlap; live entries with an id not already present are appended in the
/// order given. The result is duplicate-free by id.
///
/// Passing an empty `live` returns `curated` unchanged; passing an empty
/// `curated` returns `live` de-duplicated by id (first occurrence wins).
pub fn merge_curated_with_live(curated: Vec<ModelInfo>, live: Vec<ModelInfo>) -> Vec<ModelInfo> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut merged: Vec<ModelInfo> = Vec::with_capacity(curated.len() + live.len());

    for model in curated.into_iter().chain(live) {
        if seen.insert(model.id.clone()) {
            merged.push(model);
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::llm::{ModelCapabilities, ModelInfo};

    fn model(id: &str, ctx: u64, reasoning: bool) -> ModelInfo {
        ModelInfo::new(id)
            .with_display_name(format!("display:{id}"))
            .with_context_limit(ctx)
            .with_capabilities(ModelCapabilities {
                reasoning,
                vision: true,
                tools: true,
                embedding: false,
            })
    }

    /// Bare live entry: just an id, no metadata — mirrors what a provider's
    /// `/v1/models` endpoint returns.
    fn bare(id: &str) -> ModelInfo {
        ModelInfo::new(id)
    }

    #[test]
    fn empty_live_returns_curated_unchanged() {
        let curated = vec![model("a", 100, true), model("b", 200, false)];
        let merged = merge_curated_with_live(curated.clone(), vec![]);
        assert_eq!(merged, curated);
    }

    #[test]
    fn empty_curated_returns_live_deduped() {
        let live = vec![bare("x"), bare("y"), bare("x")];
        let merged = merge_curated_with_live(vec![], live);
        let ids: Vec<&str> = merged.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["x", "y"]);
    }

    #[test]
    fn curated_metadata_wins_on_overlap() {
        // Curated "a" has rich metadata; the live endpoint reports the same id
        // with nothing useful. The merged "a" must keep the curated metadata.
        let curated = vec![model("a", 400_000, true)];
        let live = vec![bare("a")];
        let merged = merge_curated_with_live(curated, live);
        assert_eq!(merged.len(), 1, "id overlap must not duplicate");
        assert_eq!(merged[0].context_limit, Some(400_000));
        assert!(merged[0].capabilities.reasoning);
        assert_eq!(merged[0].display_name, "display:a");
    }

    #[test]
    fn unknown_live_models_appended_after_curated() {
        let curated = vec![model("a", 100, false), model("b", 200, false)];
        let live = vec![bare("a"), bare("z"), bare("c")];
        let merged = merge_curated_with_live(curated, live);
        let ids: Vec<&str> = merged.iter().map(|m| m.id.as_str()).collect();
        // Curated order first (a, b), then the unknown live ids in endpoint
        // order (z, c). The overlapping "a" stays in its curated slot.
        assert_eq!(ids, ["a", "b", "z", "c"]);
    }

    #[test]
    fn ordering_is_stable_curated_then_live() {
        let curated = vec![model("m1", 1, false), model("m2", 2, false)];
        let live = vec![bare("m3"), bare("m4")];
        let merged = merge_curated_with_live(curated, live);
        let ids: Vec<&str> = merged.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["m1", "m2", "m3", "m4"]);
    }

    #[test]
    fn duplicate_live_ids_keep_first_occurrence() {
        let live = vec![
            model("dup", 10, true),
            model("dup", 20, false),
            bare("other"),
        ];
        let merged = merge_curated_with_live(vec![], live);
        let ids: Vec<&str> = merged.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["dup", "other"]);
        // First occurrence's metadata is the one retained.
        assert_eq!(merged[0].context_limit, Some(10));
        assert!(merged[0].capabilities.reasoning);
    }

    #[test]
    fn both_empty_returns_empty() {
        let merged = merge_curated_with_live(vec![], vec![]);
        assert!(merged.is_empty());
    }
}
