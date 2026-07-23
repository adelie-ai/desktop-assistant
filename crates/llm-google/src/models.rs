//! Curated Gemini model table, context windows, and capability inference.
//!
//! Neither the Vertex publisher-model listing nor the Gemini API `models`
//! endpoint reliably exposes context windows or capability flags, so this
//! curated table is the source of truth for the models we know about. It is
//! merged with any live listing via
//! [`merge_curated_with_live`](desktop_assistant_llm_http::merge_curated_with_live):
//! curated metadata wins on id overlap, unknown live ids are appended.

use desktop_assistant_core::ports::llm::{ModelCapabilities, ModelInfo};

/// Gemini 2.5 / 2.0 models ship a ~1M-token input window.
const CONTEXT_1M: u64 = 1_048_576;
/// Gemini 1.5 Pro ships a ~2M-token input window.
const CONTEXT_2M: u64 = 2_097_152;

/// Return the prompt-token context window for a known Gemini model id.
///
/// Returns `None` for unrecognized ids; callers fall back to the universal
/// default and/or message-count heuristics.
pub fn context_limit_for_model(model: &str) -> Option<u64> {
    if model.starts_with("gemini-2.5") || model.starts_with("gemini-2.0") {
        Some(CONTEXT_1M)
    } else if model.starts_with("gemini-1.5-pro") {
        Some(CONTEXT_2M)
    } else if model.starts_with("gemini-1.5") {
        Some(CONTEXT_1M)
    } else {
        None
    }
}

/// Whether a model supports extended thinking (`thinkingConfig`). Gemini 2.5
/// models are thinking-capable, as are the explicit `-thinking` 2.0 previews.
pub fn model_supports_thinking(model: &str) -> bool {
    model.starts_with("gemini-2.5") || model.contains("thinking")
}

/// Infer capability flags from a Gemini model id.
pub fn infer_capabilities(id: &str) -> ModelCapabilities {
    if id.to_ascii_lowercase().contains("embedding") {
        return ModelCapabilities {
            reasoning: false,
            vision: false,
            tools: false,
            embedding: true,
        };
    }
    ModelCapabilities {
        reasoning: model_supports_thinking(id),
        // Every current Gemini chat model accepts image input.
        vision: true,
        tools: true,
        embedding: false,
    }
}

/// The curated Gemini family exposed by this connector.
pub fn curated_gemini_models() -> Vec<ModelInfo> {
    fn chat(id: &str, name: &str, ctx: u64) -> ModelInfo {
        ModelInfo::new(id)
            .with_display_name(name)
            .with_context_limit(ctx)
            .with_capabilities(infer_capabilities(id))
    }
    fn embed(id: &str, name: &str) -> ModelInfo {
        ModelInfo::new(id)
            .with_display_name(name)
            .with_capabilities(infer_capabilities(id))
    }
    vec![
        chat("gemini-2.5-pro", "Gemini 2.5 Pro", CONTEXT_1M),
        chat("gemini-2.5-flash", "Gemini 2.5 Flash", CONTEXT_1M),
        chat("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", CONTEXT_1M),
        chat("gemini-2.0-flash", "Gemini 2.0 Flash", CONTEXT_1M),
        chat("gemini-1.5-pro", "Gemini 1.5 Pro", CONTEXT_2M),
        chat("gemini-1.5-flash", "Gemini 1.5 Flash", CONTEXT_1M),
        embed("text-embedding-004", "Text Embedding 004"),
        embed("gemini-embedding-001", "Gemini Embedding 001"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_limit_for_2_5_family_is_1m() {
        assert_eq!(context_limit_for_model("gemini-2.5-pro"), Some(CONTEXT_1M));
        assert_eq!(
            context_limit_for_model("gemini-2.5-flash"),
            Some(CONTEXT_1M)
        );
        assert_eq!(
            context_limit_for_model("gemini-2.5-flash-lite"),
            Some(CONTEXT_1M)
        );
    }

    #[test]
    fn context_limit_for_1_5_pro_is_2m() {
        assert_eq!(context_limit_for_model("gemini-1.5-pro"), Some(CONTEXT_2M));
    }

    #[test]
    fn context_limit_for_1_5_flash_is_1m() {
        assert_eq!(
            context_limit_for_model("gemini-1.5-flash"),
            Some(CONTEXT_1M)
        );
    }

    #[test]
    fn context_limit_for_unknown_model_is_none() {
        assert_eq!(context_limit_for_model("gpt-5"), None);
        assert_eq!(context_limit_for_model("totally-made-up"), None);
    }

    #[test]
    fn thinking_supported_for_2_5_family() {
        assert!(model_supports_thinking("gemini-2.5-pro"));
        assert!(model_supports_thinking("gemini-2.5-flash"));
        assert!(model_supports_thinking("gemini-2.0-flash-thinking-exp"));
    }

    #[test]
    fn thinking_not_supported_for_1_5_or_unknown() {
        assert!(!model_supports_thinking("gemini-1.5-pro"));
        assert!(!model_supports_thinking("gemini-1.5-flash"));
        assert!(!model_supports_thinking("gpt-5"));
    }

    #[test]
    fn curated_table_is_non_empty_and_has_flagship() {
        let models = curated_gemini_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id == "gemini-2.5-pro"));
        assert!(models.iter().any(|m| m.id == "gemini-2.5-flash"));
    }

    #[test]
    fn curated_2_5_models_flagged_reasoning_and_tools() {
        let models = curated_gemini_models();
        let pro = models.iter().find(|m| m.id == "gemini-2.5-pro").unwrap();
        assert!(pro.capabilities.reasoning, "2.5 pro is a reasoning model");
        assert!(pro.capabilities.tools);
        assert!(pro.capabilities.vision);
        assert!(!pro.capabilities.embedding);
        assert_eq!(pro.context_limit, Some(CONTEXT_1M));
    }

    #[test]
    fn curated_embedding_model_flagged_embedding() {
        let models = curated_gemini_models();
        let embed = models
            .iter()
            .find(|m| m.capabilities.embedding)
            .expect("curated table includes an embedding model");
        assert!(!embed.capabilities.tools);
        assert!(!embed.capabilities.reasoning);
    }

    #[test]
    fn infer_capabilities_marks_embedding_models() {
        let caps = infer_capabilities("text-embedding-004");
        assert!(caps.embedding);
        assert!(!caps.tools);
    }

    #[test]
    fn infer_capabilities_marks_2_5_reasoning() {
        let caps = infer_capabilities("gemini-2.5-flash");
        assert!(caps.reasoning);
        assert!(caps.tools);
        assert!(caps.vision);
        assert!(!caps.embedding);
    }
}
