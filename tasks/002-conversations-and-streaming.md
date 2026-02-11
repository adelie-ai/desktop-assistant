# 002 — Conversations and Streaming Prompt/Response

## Background

The scaffolded workspace has no real functionality. The assistant needs to manage conversations and stream LLM responses so that a D-Bus client can interactively chat with an LLM backend.

## Change Description

- Add domain types: `ConversationId`, `Conversation`, `Message`, `Role`, `ConversationSummary`.
- Add `CoreError` variants: `ConversationNotFound`, `Llm`, `Storage`.
- Define outbound port traits: `LlmClient` (callback-based streaming), `ConversationStore` (CRUD).
- Define inbound port trait: `ConversationService` (create, list, get, delete, send_prompt).
- Implement `ConversationHandler<S, L>` in core — generic over store and LLM, with injected ID generator.
- Implement `InMemoryConversationStore` in daemon (Mutex + HashMap).
- Create `crates/llm-openai` crate with `OpenAiClient` — reqwest SSE streaming against OpenAI ChatCompletions API.
- Create D-Bus adapter (`org.desktopAssistant.Conversations`) with methods for CRUD and `SendPrompt`, plus signals `ResponseChunk`, `ResponseComplete`, `ResponseError`.
- Wire everything in daemon `main.rs`: build store, LLM client, service, register D-Bus objects.
- 60 tests: unit tests per module + integration tests covering full lifecycle and streaming abort.

## Expected Behavior

- `cargo build` succeeds with no errors.
- `cargo test` passes all 60 tests.
- Running the daemon with `OPENAI_API_KEY` set registers on the session bus.
- D-Bus clients can call `CreateConversation`, `SendPrompt`, and receive streaming `ResponseChunk` signals followed by `ResponseComplete`.
