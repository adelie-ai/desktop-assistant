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

use desktop_assistant_core::ports::llm::{ModelCapabilities, ModelInfo, ModelKind};
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
    #[serde(default)]
    openrouter: Vec<DefaultEntry>,
    #[serde(default)]
    azure: Vec<DefaultEntry>,
    #[serde(default)]
    google: Vec<DefaultEntry>,
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
            // The curated defaults are hand-authored and always known: an
            // `embedding = true` entry is an embedding model, everything else
            // is generative. There is no `Unknown` in this table.
            kind: if entry.capabilities.embedding {
                ModelKind::Embedding
            } else {
                ModelKind::Generative
            },
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
        "openrouter" => &file.openrouter,
        "azure" => &file.azure,
        "google" => &file.google,
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
                .any(|m| m.id == "mxbai-embed-large:335m" && m.capabilities.is_embedding()),
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

    #[test]
    fn defaults_load_for_new_cloud_connectors() {
        // OpenRouter and Google seed the picker with common chat ids.
        assert!(
            defaults_for("openrouter")
                .iter()
                .any(|m| m.id == "anthropic/claude-sonnet-4-6"),
            "openrouter defaults missing"
        );
        assert!(
            defaults_for("google")
                .iter()
                .any(|m| m.id == "gemini-2.5-pro" && m.capabilities.reasoning),
            "google defaults missing or mis-tagged"
        );
        // Azure carries curated base-model metadata (not directly selectable).
        assert!(
            defaults_for("azure")
                .iter()
                .any(|m| m.id == "text-embedding-3-small" && m.capabilities.is_embedding()),
            "azure embedding default missing or mis-tagged"
        );
    }

    #[tokio::test]
    async fn every_connector_reports_a_model_kind() {
        use crate::connections::Connector;
        use desktop_assistant_core::ports::llm::{LlmClient, ModelKind};

        // Exhaustive over every connector: the `match` has no wildcard, so a new
        // `Connector` variant cannot compile until it is handled here, forcing
        // the author to decide how it classifies models (#647).
        //
        // The catalog is fetched the cheapest offline way per connector: the
        // two curated-only HTTP connectors list from their in-memory table;
        // the rest fall back to their bundled defaults. Bedrock classifies from
        // live AWS output-modality metadata, which cannot be reached offline --
        // that path is proven by `bedrock_derives_kind_from_output_modalities`
        // in `llm-bedrock`, so its arm has no offline catalog to assert.
        for connector in [
            Connector::Ollama,
            Connector::Anthropic,
            Connector::Bedrock,
            Connector::OpenAi,
            Connector::OpenRouter,
            Connector::Azure,
            Connector::Google,
        ] {
            let catalog: Vec<ModelInfo> = match connector {
                Connector::OpenAi => desktop_assistant_llm_openai::OpenAiClient::new("k".into())
                    .list_models()
                    .await
                    .unwrap_or_default(),
                Connector::Anthropic => {
                    desktop_assistant_llm_anthropic::AnthropicClient::new("k".into())
                        .list_models()
                        .await
                        .unwrap_or_default()
                }
                Connector::Ollama => defaults_for("ollama"),
                Connector::OpenRouter => defaults_for("openrouter"),
                Connector::Azure => defaults_for("azure"),
                Connector::Google => defaults_for("google"),
                Connector::Bedrock => Vec::new(),
            };

            // No connector may surface a model it left `Unknown`.
            for m in &catalog {
                assert_ne!(
                    m.capabilities.kind,
                    ModelKind::Unknown,
                    "{connector:?} left {} classified Unknown",
                    m.id
                );
                // Where the id follows the near-universal embed convention, the
                // classified kind must agree.
                if m.id.to_ascii_lowercase().contains("embed") {
                    assert_eq!(
                        m.capabilities.kind,
                        ModelKind::Embedding,
                        "{connector:?} model {} looks like an embedding model but wasn't classified one",
                        m.id
                    );
                }
            }

            // The connectors with an offline catalog must actually classify
            // something -- a generative model at minimum -- so the arm proves
            // real classification rather than passing vacuously.
            if !matches!(connector, Connector::Bedrock) {
                assert!(
                    !catalog.is_empty(),
                    "{connector:?} should expose a classifiable catalog offline"
                );
                assert!(
                    catalog
                        .iter()
                        .any(|m| m.capabilities.kind == ModelKind::Generative),
                    "{connector:?} should classify at least one generative model"
                );
            }
        }
    }
}
