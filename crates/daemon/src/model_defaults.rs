//! Curated model defaults shipped with the daemon.
//!
//! `data/model_defaults.toml` is embedded at compile time and parsed once on
//! demand. Connectors whose live `list_models()` doesn't return a particular
//! id can fall back to these entries so the UI sees a reasonable default
//! list — important for Ollama (only returns locally pulled models) and for
//! API-less connectors that don't expose a model enumeration endpoint.
//!
//! Live results always win; defaults only fill in ids that the live response
//! didn't include.

use std::collections::HashMap;
use std::sync::OnceLock;

use desktop_assistant_core::ports::llm::{ModelCapabilities, ModelInfo};
use serde::Deserialize;

const DEFAULTS_TOML: &str = include_str!("../data/model_defaults.toml");

#[derive(Debug, Clone, Deserialize)]
struct DefaultsFile {
    #[serde(default)]
    ollama: Vec<DefaultEntry>,
    #[serde(default)]
    anthropic: Vec<DefaultEntry>,
    #[serde(default)]
    openai: Vec<DefaultEntry>,
    #[serde(default)]
    bedrock: Vec<DefaultEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct DefaultEntry {
    id: String,
    display_name: String,
    #[serde(default)]
    context_limit: Option<u64>,
    #[serde(default)]
    capabilities: DefaultCapabilities,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct DefaultCapabilities {
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    vision: bool,
    #[serde(default)]
    tools: bool,
    #[serde(default)]
    embedding: bool,
}

impl From<DefaultEntry> for ModelInfo {
    fn from(entry: DefaultEntry) -> Self {
        let mut info = ModelInfo::new(entry.id).with_display_name(entry.display_name);
        if let Some(ctx) = entry.context_limit {
            info = info.with_context_limit(ctx);
        }
        info.with_capabilities(ModelCapabilities {
            reasoning: entry.capabilities.reasoning,
            vision: entry.capabilities.vision,
            tools: entry.capabilities.tools,
            embedding: entry.capabilities.embedding,
        })
    }
}

fn parsed() -> &'static DefaultsFile {
    static CACHE: OnceLock<DefaultsFile> = OnceLock::new();
    CACHE.get_or_init(|| {
        toml::from_str(DEFAULTS_TOML).expect("model_defaults.toml is malformed at compile time")
    })
}

/// Curated defaults for a connector type (`"ollama"`, `"openai"`,
/// `"anthropic"`, `"bedrock"`). Returns an empty vector for unknown types.
pub fn defaults_for(connector_type: &str) -> Vec<ModelInfo> {
    let file = parsed();
    let raw = match connector_type {
        "ollama" => &file.ollama,
        "anthropic" => &file.anthropic,
        "openai" => &file.openai,
        "bedrock" => &file.bedrock,
        _ => return Vec::new(),
    };
    raw.iter().cloned().map(Into::into).collect()
}

/// Merge a live `list_models()` result with the curated defaults for the
/// given connector type. Entries from `live` win over defaults with the same
/// id; defaults only fill in ids that aren't already present.
pub fn merge_with_defaults(connector_type: &str, mut live: Vec<ModelInfo>) -> Vec<ModelInfo> {
    let defaults = defaults_for(connector_type);
    if defaults.is_empty() {
        return live;
    }

    let live_ids: HashMap<String, ()> = live.iter().map(|m| (m.id.clone(), ())).collect();
    for entry in defaults {
        if !live_ids.contains_key(&entry.id) {
            live.push(entry);
        }
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_load_for_ollama() {
        let entries = defaults_for("ollama");
        assert!(!entries.is_empty(), "expected at least one ollama default");
        assert!(
            entries
                .iter()
                .any(|m| m.id == "mxbai-embed-large:335m" && m.capabilities.embedding),
            "embedding default missing or mis-tagged"
        );
    }

    #[test]
    fn merge_keeps_live_metadata() {
        let live = vec![
            ModelInfo::new("llama3.2:3b")
                .with_display_name("LIVE override")
                .with_context_limit(100),
        ];
        let merged = merge_with_defaults("ollama", live);
        let llama = merged.iter().find(|m| m.id == "llama3.2:3b").unwrap();
        assert_eq!(llama.display_name, "LIVE override");
        assert_eq!(llama.context_limit, Some(100));
        // Defaults still bring in non-conflicting ids.
        assert!(merged.iter().any(|m| m.id == "mxbai-embed-large:335m"));
    }

    #[test]
    fn unknown_connector_returns_empty() {
        assert!(defaults_for("nonexistent").is_empty());
    }
}
