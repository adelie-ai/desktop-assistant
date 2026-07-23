# OpenRouter Connector

Crate: `desktop-assistant-llm-openrouter` (planned)

OpenRouter is an OpenAI-compatible aggregator that routes one API to many model
vendors (`anthropic/*`, `openai/*`, `google/*`, `meta-llama/*`, ...). Built on the
OpenAI Chat Completions dialect that begins as an internal module of this crate
and is later extracted to `llm-openai-compat` (see `cloud-connector-abstraction.md`).

## API Details

- Endpoint: `{base_url}/chat/completions` (POST, streaming SSE)
- Default base URL: `https://openrouter.ai/api/v1`
- Auth: `Authorization: Bearer {OPENROUTER_API_KEY}`
- Attribution headers (optional): `HTTP-Referer` / `X-Title`. Sent as a fixed,
  public Adele identifier; validated (reject CR/LF/control chars, cap length);
  never derived from user, session, or internal-URL data. Not exposed as
  per-connection config in v1.
- Models endpoint: `{base_url}/models` (rich metadata: `context_length`,
  `pricing`, `supported_parameters`, `architecture`).
- Default model: `anthropic/claude-sonnet-4-6` - verify the exact `vendor/model`
  slug against the live `/models` catalog at ship time (slugs are version-fragile;
  a stale default is recoverable via the picker).

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Environment | `OPENROUTER_API_KEY` | Yes |
| Environment | `OPENROUTER_MODEL` | No |
| Environment | `OPENROUTER_BASE_URL` | No |
| Config file | `daemon.toml` `[connections.<id>]` `type = "openrouter"` | No |

`OpenRouterConnection` mirrors `OpenAiConnection` (base_url, api_key_env, secret,
connect/stream timeouts, max_context_tokens). Env-var names derive automatically
from the connector key `openrouter`. The struct holds the api key and implements a
redacting `Debug` (render `<redacted; len=N>`), per the uniform contract. Its
fields fit the existing OpenAI-shaped `ConnectionConfigView`, so OpenRouter is
GUI-configurable from the start.

## Model addressing

The logical model id is a `vendor/model` string in the request body `model` field;
no URL rewriting. The curated table lists a small current set of `vendor/model`
ids with capability flags; the live `/models` endpoint fills the long tail
(merged via `merge_curated_with_live`, behind the shared TTL model cache).
`context_limit` comes from the live `context_length`.

## Request / response mapping

Standard OpenAI Chat Completions via the compat module:

- System `Message` -> `messages[{role:"system"}]`; User/Assistant ->
  `messages[{role, content}]`; assistant tool calls -> `tool_calls[]` with
  `function.name` / `function.arguments`; tool results ->
  `messages[{role:"tool", tool_call_id, content}]`.
- `ToolDefinition` -> `tools[{type:"function", function:{name, description,
  parameters}}]`, with the parameters JSON Schema passed through the shared
  sanitizer (see Robustness).
- Streaming: SSE `data:` frames, `choices[0].delta.content` for text,
  `choices[0].delta.tool_calls[]` (indexed) accumulated with
  `ToolCallAccumulator<usize>`; `[DONE]` terminates.
- Request `usage` accounting so the final chunk carries token details including
  cache activity - use OpenRouter's documented mechanism (top-level
  `usage: {include: true}`), not the OpenAI `stream_options` form; the compat
  module reconciles the two shapes.

## Prompt caching

Unified and cross-provider. OpenRouter accepts `cache_control` block markers and
translates them per routed provider: an Anthropic-style `cache_control` marker
becomes an OpenAI `prompt_cache_breakpoint` when routed to a supporting OpenAI
model, and a default (5-minute) cache when routed to Anthropic or Google.
OpenAI/DeepSeek/Grok-family routes also cache automatically without a marker.

Policy: mark the system block with `cache_control` (the tool list is dynamic, so
never mark tools/messages - see `anthropic.md`) and let OpenRouter normalize for
whatever it routes to. Read cache activity from `usage.prompt_tokens_details`:
`cache_write_tokens` -> `cache_creation_input_tokens`, `cached_tokens` ->
`cache_read_input_tokens`.

The `cache_control`-into-content-array mechanic is the shared compat helper; the
decision to mark is OpenRouter-local. Follow-up worth noting: OpenRouter accepts a
`session_id` sticky-routing key that pins a conversation to one upstream to
maximize cache hits - map it to the conversation id in a later iteration.

## Hosted tool search

Off in v1. Namespaces flatten into the standard `tools` array via the trait's
default. `supports_hosted_tool_search()` -> `false`.

## Reasoning

Unified field: `reasoning: { effort: "low|medium|high" }` or
`reasoning: { max_tokens: N }`; OpenRouter normalizes per routed model. To emit
both shapes end-to-end, OpenRouter needs its **own** arm in the daemon's
`map_effort_to_reasoning_config` (not the shared `openai` arm, which only ever
sets `reasoning_effort` - the `max_tokens` branch would be dead code there). Map
`reasoning_effort` -> `effort` and `thinking_budget_tokens` -> `max_tokens`;
include the field only when the config is non-empty. If a dedicated arm is not
worth it in v1, drop the `max_tokens` mapping and support `effort` only.

## Robustness (shared compat module)

- **Tool-schema sanitization**: strip top-level `oneOf/anyOf/allOf`, inject a
  `type` when missing (one bad MCP schema otherwise 400s the whole turn - the
  Bedrock #214/#67 failure class). OpenRouter's long tail proxies to strict
  backends, so this is not optional.
- **Empty-key tool input `{"":{}}`**: normalize on the history path; gpt-oss (which
  OpenRouter routes) emits it, and re-sending it 400s every subsequent turn.
- **Streaming-with-tools-unsupported** (#619): some routed backends accept tools
  only on a non-streaming request. `detect_streaming_tools_unsupported` (a narrow
  connector-boundary match requiring *tools* + *streaming* + a negation, so it
  does not false-positive on an unrelated error) classifies that provider error
  to `CoreError::ToolsUnsupported`. On it, `stream_completion` retries once via a
  non-streaming `/chat/completions` dispatch (`stream: false`, parsed by
  `parse_chat_completion`, full text emitted through `on_chunk` once) and records
  the model in a per-connection memo so the next tools turn skips the stream
  attempt. A plain tools-unsupported error (not streaming-specific) is excluded -
  non-streaming would not help, so it surfaces. A non-streaming failure surfaces
  as-is and never loops back to streaming. Mirrors the Bedrock pattern (#67).

## Error mapping

OpenAI-shaped envelope (`{error:{code,message,type}}`), classifiers lenient
across upstreams and unit-tested against captured fixtures:

- 429 -> `RateLimited` (with `Retry-After`).
- 402 / insufficient credits / quota -> `QuotaExceeded`.
- `context_length_exceeded` / "maximum context" -> `ContextOverflow` (parse counts
  when present; OpenAI `(max, prompt)` order, swapped to `(prompt, max)`).
- 5xx / 503 -> `RateLimited`.
- else -> a clear `Llm` message; never a raw hang.

## Embeddings

OpenRouter's embedding coverage is limited and model-specific. v1 does not
implement `EmbeddingClient`; `Connector::supports_embeddings` returns `false`
explicitly (the enum default is `true`), and the embeddings availability gate must
exclude OpenRouter so an embedding purpose does not silently build an OpenAI-shaped
client against `/embeddings`.

## Test plan

- `convert_messages` round-trip; flat `function` tool serialization; tool-call
  accumulation from indexed `delta.tool_calls`.
- Schema + empty-key sanitization (shared compat tests).
- Caching: system-block `cache_control` present; `cache_write_tokens` /
  `cached_tokens` parsed into usage.
- Reasoning: `effort` and `max_tokens` mapping; omitted when empty.
- Error paths (httpmock): 400 overflow, 402 credits, 429 + retry, 5xx,
  tools-unsupported.
- Streaming-with-tools-unsupported fallback (#619): the provider error is
  classified to `ToolsUnsupported`; the non-streaming retry returns the tool
  calls + usage + text-via-`on_chunk`; the memo skips streaming on the second
  call (the streaming endpoint is not hit again); a model that does not error
  stays on streaming; a non-streaming failure surfaces without looping;
  cancellation is honoured on the non-streaming path.
- `MODEL_OVERRIDE` routes the body `model`.
- Redacting `Debug`; cancellation mid-stream; malformed-SSE tolerance;
  callback-abort preserves accumulated tool calls and usage.

## Open questions

- Default attribution `HTTP-Referer` / `X-Title` value (Adele project URL).
- Whether to expose provider-routing preferences (`provider: {order,
  allow_fallbacks}`) as config, or keep v1 minimal.
