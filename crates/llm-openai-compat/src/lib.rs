//! OpenAI Chat Completions dialect: shared building blocks for the connectors
//! that speak it (OpenRouter, Azure OpenAI).
//!
//! This crate is a **library of building blocks, not an `LlmClient`**. The
//! `llm-openrouter` and `llm-azure` connectors each assemble their own
//! top-level request envelope (they differ: Azure uses `max_completion_tokens`
//! for reasoning models, OpenRouter uses a `reasoning` object, and so on) and
//! own their auth headers, base-URL shaping, and model addressing. What they
//! share -- and what lives here -- is the wire dialect itself:
//!
//! - [`to_chat_messages`] / [`ChatMessage`]: domain [`Message`]s ->
//!   `messages[]` in the `chat/completions` shape (`system` / `user` /
//!   `assistant` with optional `tool_calls` / `tool` results keyed by
//!   `tool_call_id`).
//! - [`to_chat_tools`] / [`ChatTool`]: domain [`ToolDefinition`]s ->
//!   `tools[{type:"function", function:{name,description,parameters}}]`, with
//!   the parameter schema passed through [`sanitize_tool_schema`].
//! - [`sanitize_tool_schema`] / [`sanitize_tool_arguments`]: the two
//!   robustness fixups the Bedrock connector proved necessary (strip top-level
//!   `oneOf`/`anyOf`/`allOf`; drop the `{"":{}}` empty-key garbage gpt-oss
//!   emits) so one bad MCP schema or tool call cannot 400 the whole turn.
//! - [`consume_chat_stream`] / [`parse_chat_chunk`] / [`ChatChunk`]: SSE
//!   `choices[].delta` parsing with indexed `tool_calls` accumulation and a
//!   final `usage` read, terminating on `[DONE]`.
//! - [`parse_usage`]: `usage` -> [`TokenUsage`], including the cache-activity
//!   fields (`prompt_tokens_details.cached_tokens` / `.cache_write_tokens`).
//! - [`mark_system_cache_breakpoint`]: add a `cache_control` marker to the last
//!   system message via the multi-part content-array form.
//! - [`classify_error`] plus the [`detect_context_overflow`] /
//!   [`detect_insufficient_quota`] sub-detectors: the base OpenAI-compatible
//!   error mapping, which connectors extend (OpenRouter adds 402-credits, Azure
//!   adds `content_filter`) by wrapping these.
//!
//! Streaming reuses the shared `llm-http` primitives
//! ([`next_step`](desktop_assistant_llm_http::next_step),
//! [`StreamStep`](desktop_assistant_llm_http::StreamStep),
//! [`build_response`](desktop_assistant_llm_http::build_response),
//! [`parse_retry_after_header`](desktop_assistant_llm_http::parse_retry_after_header)),
//! so the connect race and stall loop stay in one place.

mod errors;
mod messages;
mod nonstreaming;
mod streaming;
mod tools;
mod usage;

pub use errors::{
    ContextOverflowInfo, classify_error, detect_context_overflow, detect_insufficient_quota,
    detect_streaming_tools_unsupported,
};
pub use nonstreaming::{dispatch_non_streaming, parse_chat_completion};
pub use messages::{
    CacheControl, ChatContent, ChatContentPart, ChatFunctionCall, ChatMessage, ChatToolCall,
    mark_system_cache_breakpoint, sanitize_tool_arguments, to_chat_messages,
};
pub use streaming::{
    ChatChoice, ChatChunk, ChatDelta, ChatFunctionDelta, ChatToolCallDelta, consume_chat_stream,
    parse_chat_chunk,
};
pub use tools::{ChatTool, ChatToolFunction, sanitize_tool_schema, to_chat_tools};
pub use usage::parse_usage;
