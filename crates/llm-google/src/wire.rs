//! Serde types for the Gemini `generateContent` wire schema.
//!
//! These model the request body (`contents`, `systemInstruction`, `tools`,
//! `generationConfig`) and the streamed `GenerateContentResponse` frames
//! (`candidates[].content.parts[]`, `promptFeedback`, `usageMetadata`). The
//! same schema is served by both the Vertex AI surface and the Gemini API
//! (AI Studio) surface; only the host, path prefix, and auth differ, so these
//! types are shared across `auth_mode`.
//!
//! Field names follow Gemini's camelCase JSON (`systemInstruction`,
//! `functionCall`, `maxOutputTokens`, …); serde `rename`/`rename_all` bridges
//! them to Rust's snake_case idiom.

use serde::{Deserialize, Serialize};

/// Top-level `generateContent` request body.
#[derive(Serialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

/// The concatenated system prompt, sent out-of-band from the turn `contents`.
#[derive(Serialize, Debug, Default)]
pub struct SystemInstruction {
    pub parts: Vec<Part>,
}

/// One turn in the conversation. `role` is `"user"` or `"model"`; the system
/// prompt never appears here (it rides in [`SystemInstruction`]).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// A single part of a turn: text, a model-emitted `functionCall`, or a
/// caller-supplied `functionResponse`. Exactly one field is populated.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// Gemini flags a thinking-trace text part with `thought: true`. We read
    /// it to keep thought text out of the user-visible stream.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub thought: bool,
}

impl Part {
    /// A plain text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            function_call: None,
            function_response: None,
            thought: false,
        }
    }

    /// A model `functionCall` part (arguments as a JSON object).
    pub fn function_call(name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            text: None,
            function_call: Some(FunctionCall {
                name: name.into(),
                args,
            }),
            function_response: None,
            thought: false,
        }
    }

    /// A `functionResponse` part (`response` MUST be a JSON object).
    pub fn function_response(name: impl Into<String>, response: serde_json::Value) -> Self {
        Self {
            text: None,
            function_call: None,
            function_response: Some(FunctionResponse {
                name: name.into(),
                response,
            }),
            thought: false,
        }
    }
}

/// A tool-call request: the function name plus its arguments as a JSON object.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

/// A tool-call result: the function name plus its result as a JSON object.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

/// A `tools` entry. Gemini flattens tool declarations under a single
/// `functionDeclarations` array.
#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// A single callable function's contract.
#[derive(Serialize, Debug)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    /// Sanitized OpenAPI-subset schema. Omitted entirely when the tool takes
    /// no parameters (an empty schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Generation knobs. `thinkingConfig` is included only for thinking-capable
/// models when a positive budget is requested.
#[derive(Serialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

impl GenerationConfig {
    /// True when no field is populated — the whole `generationConfig` object
    /// should then be omitted from the request.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.max_output_tokens.is_none()
            && self.thinking_config.is_none()
    }
}

/// Extended-thinking configuration (`generationConfig.thinkingConfig`).
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    pub thinking_budget: u32,
    pub include_thoughts: bool,
}

// --- Response types --------------------------------------------------------

/// One streamed `GenerateContentResponse` frame.
#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
}

/// A single candidate completion.
#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// Prompt-level feedback; a non-empty `blockReason` means the *prompt* was
/// refused by the safety filter before any generation.
#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[serde(default)]
    pub block_reason: Option<String>,
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// A per-category safety score. `blocked` marks the category that tripped.
#[derive(Deserialize, Debug, Default, Clone)]
pub struct SafetyRating {
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub blocked: bool,
}

/// Token accounting. `promptTokenCount` -> input, `candidatesTokenCount` ->
/// output, `cachedContentTokenCount` -> cache-read. `thoughtsTokenCount` is
/// carried for observability only.
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: Option<u64>,
    #[serde(default)]
    pub candidates_token_count: Option<u64>,
    #[serde(default)]
    pub cached_content_token_count: Option<u64>,
    /// Reasoning-trace tokens, surfaced for observability only.
    #[serde(default)]
    pub thoughts_token_count: Option<u64>,
}

/// Google error envelope: `{ "error": { "code", "message", "status" } }`.
#[derive(Deserialize, Debug, Default)]
pub struct ErrorEnvelope {
    #[serde(default)]
    pub error: ErrorBody,
}

/// The inner error object. `status` is the canonical code
/// (`RESOURCE_EXHAUSTED`, `INVALID_ARGUMENT`, `UNAVAILABLE`, …).
#[derive(Deserialize, Debug, Default)]
pub struct ErrorBody {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}
