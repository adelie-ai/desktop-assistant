# Google (Vertex AI Gemini) Connector

Crate: `desktop-assistant-llm-google` (planned)

Google's Bedrock equivalent is **Vertex AI**: a cloud-credential-authenticated,
project- and region-scoped gateway hosting Gemini. This connector targets the
native **Gemini `generateContent`** schema - the most distinct of the three new
connectors, closest in structure to `llm-bedrock`. See
`cloud-connector-abstraction.md`.

## API Details

Verified current (2026-07-22): Vertex `v1` REST is the live GA surface. The
product branding is migrating to "Gemini Enterprise Agent Platform" but the REST
path is unchanged.

- Endpoint (streaming):
  `POST {base_url}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse`
- Non-streaming / resources: `:generateContent`, `cachedContents`, etc.
- `base_url`: `https://{location}-aiplatform.googleapis.com`, composed by the
  connector from `location` (so setting only `location` is sufficient and
  consistent). `location` may be a region (`us-central1`) or `global` for models
  that support it.
- Auth: OAuth2 bearer from a GCP service account or Application Default
  Credentials, resolved and refreshed by a `TokenProvider` seam. Token scope
  `https://www.googleapis.com/auth/cloud-platform`.
- Default model: `gemini-2.5-pro` (cheaper backend default `gemini-2.5-flash`).
- Default location: `us-central1`.
- API version: `v1` is GA; `v1beta1` carries the newest features and may be
  required for `thinkingConfig` - confirm which version exposes it at
  implementation and pin accordingly (kept as an internal constant, easy to bump).

### Gemini API (AI Studio) variant

Same crate, `auth_mode = "api_key"`, switching host + auth:

- `base_url`: `https://generativelanguage.googleapis.com`
- Endpoint: `/v1beta/models/{model}:streamGenerateContent?alt=sse`
- Auth: `GOOGLE_API_KEY` via the `x-goog-api-key` **header** (never `?key=` in the
  URL - a query key leaks into logs, tracing, and error strings).

Both surfaces speak the same `generateContent` schema; only host, path prefix, and
auth differ.

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Environment | `GOOGLE_APPLICATION_CREDENTIALS` (service-account JSON path) or ADC | Vertex |
| Environment | `GOOGLE_API_KEY` | Gemini API mode |
| Environment | `GOOGLE_CLOUD_PROJECT` (fallback `GOOGLE_PROJECT`) | Vertex |
| Environment | `GOOGLE_CLOUD_LOCATION` (fallback `GOOGLE_LOCATION`) | No |
| Environment | `GOOGLE_MODEL` | No |
| Config file | `daemon.toml` `[connections.<id>]` `type = "google"` | Recommended |

`GoogleConnection` uses **first-class** fields threaded through `ResolvedLlmConfig`
and dedicated builder setters (`with_project` / `with_location` / `with_auth_mode`
/ `with_credentials_path`), following the Bedrock precedent - the `new(api_key)`
slot is NOT overloaded for a credentials path. It carries `project`, `location`,
`auth_mode` (`vertex` | `api_key`), `credentials_path` (optional; else ADC),
`api_key_env` / secret (Gemini API mode), base_url override, timeouts,
max_context_tokens. Read the standard `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`
env names (with the shorter forms as fallback) so an existing `gcloud`/ADC setup is
picked up.

Credential handling: the connector holds an OAuth access token or service-account
material and implements a redacting `Debug` (reuse the `mcp-client::oauth`
`TokenSet` type, which already redacts). The `credentials_path` is a filesystem
path (not a secret) and is not routed through the secret store; the SA JSON
contents are never read into any api-model view or error string, and the bearer
token never appears in a URL, log, or error message. Any new GCP auth crate is
gated on a `cve-mcp` / `cargo audit` scan; prefer minting the token by reusing the
workspace `jsonwebtoken` (SA JWT -> token exchange at `oauth2.googleapis.com/token`)
over a vendor SDK.

These fields exceed the current wire views and client dialogs, so **v1 Google is
config-file-only** with a documented `daemon.toml` block in `cloud-providers.md`.

## Preflight

Vertex requires a resolvable credential (ADC present or a readable
`credentials_path`), a `project`, and a `location`; Gemini-API mode requires
`GOOGLE_API_KEY`. Missing any yields a specific `Unavailable { reason }` (for
example "Vertex needs a GCP project; set GOOGLE_CLOUD_PROJECT or project=") rather
than failing opaquely at request time.

## Model addressing

The model id is a URL path segment. `MODEL_OVERRIDE` selects it. The curated table
lists the current Gemini family with context windows (1M-2M) and capability flags;
a live listing (`publishers/google/models`) fills the tail, behind the shared TTL
model cache, degrading to curated on failure.

## Request / response mapping

Full custom mapping to the Gemini schema (top-level request fields: `contents`,
`systemInstruction`, `tools`, `generationConfig`, `safetySettings`,
`cachedContent`):

- System `Message` -> `systemInstruction: { parts: [{text}] }` (concatenated).
- User -> `contents[{role:"user", parts:[{text}]}]`.
- Assistant text -> `contents[{role:"model", parts:[{text}]}]`.
- Assistant tool call -> `parts:[{functionCall:{name, args}}]` (args parsed from
  the domain arguments string into a JSON object).
- Tool result -> `contents[{role:"user", parts:[{functionResponse:{name,
  response}}]}]`, where `response` MUST be a JSON object (wrap a string result as
  `{ "result": "..." }`). Consecutive tool results merge into one `user` turn
  (the Bedrock message-merging pattern).
- `ToolDefinition` -> `tools:[{functionDeclarations:[{name, description,
  parameters}]}]`, with `parameters` passed through a **Gemini-specific** schema
  sanitizer. Gemini accepts an OpenAPI-3.0 subset and rejects more than Bedrock's
  Converse (also `$schema`, `additionalProperties`, `$ref`, some `format` values),
  so this sanitizer is reimplemented using Bedrock's `sanitize_tool_schema` as a
  guide rather than shared.
- Generation knobs -> `generationConfig: { temperature, topP, maxOutputTokens,
  thinkingConfig }`.
- Streaming: SSE (`alt=sse`), each `data:` frame a `GenerateContentResponse` with
  `candidates[0].content.parts[]` (text or `functionCall`) and a trailing
  `usageMetadata`. Text parts append + `on_chunk`; each `functionCall` part is a
  whole call (accumulate as start+finalize, not delta-append), keyed by part
  index. Take the last `usageMetadata` seen.

## Prompt caching

Implicit caching is on by default for Gemini 2.5+; no annotation. v1 relies on it
and reports `usageMetadata.cachedContentTokenCount` -> `cache_read_input_tokens`.
Explicit `cachedContents` resources (create-with-TTL, reference via `cachedContent`;
90% discount) are a documented follow-up, worthwhile only for large, stable,
reused prefixes - Adele's cache prefix (system prompt) is small and the tool list
is dynamic, so v1 does not manage explicit cache lifecycles.

## Usage fields

Map `usageMetadata`: `promptTokenCount` -> `input_tokens`, `candidatesTokenCount`
-> `output_tokens`, `cachedContentTokenCount` -> `cache_read_input_tokens`
(`thoughtsTokenCount` is available for observability).

## Hosted tool search

Not applicable; Gemini uses function declarations.
`supports_hosted_tool_search()` -> `false`; namespaces flatten. (Gemini's own
server-side tools like Search grounding are out of scope for v1.)

## Reasoning

`generationConfig.thinkingConfig: { thinkingBudget, includeThoughts }` on 2.5
models. Gemini has no effort literal, so Google gets its **own** daemon arm in
`map_effort_to_reasoning_config` that populates `thinking_budget_tokens` with a
**Gemini-calibrated** Effort -> budget table (NOT the Claude
`map_anthropic_thinking_budget` values). The connector maps `thinking_budget_tokens`
-> `thinkingBudget`, and includes `thinkingConfig` only when non-empty and the
model supports thinking.

## Error mapping and declines

Google envelope (`{error:{code,status,message}}`, `status` like
`RESOURCE_EXHAUSTED`, `INVALID_ARGUMENT`):

- 429 / `RESOURCE_EXHAUSTED` -> `RateLimited` (with `Retry-After` / `RetryInfo`).
- hard quota (billing) -> `QuotaExceeded`.
- `INVALID_ARGUMENT` with token/context phrasing -> `ContextOverflow` (parse counts
  when present).
- 401/403 (bad credentials / disabled API / wrong project) -> a clear `Llm`
  message naming the fix, never echoing the token or SA material.
- 5xx / `UNAVAILABLE` -> `RateLimited`.
- Safety block (`promptFeedback.blockReason` / `finishReason: "SAFETY"`) -> a
  business decline: non-error, non-retryable, surfaced with a specific informative
  reason (which safety category, that the request was refused), logged at
  info/debug, never dumping the flagged-content body.

## Embeddings

Vertex and the Gemini API serve embeddings with **different** shapes: Vertex
`:predict` (`instances` / `predictions`) vs Gemini API `:embedContent`. The
`EmbeddingClient` branches on `auth_mode`. This is not OpenAI-compatible, so the
embeddings factory in `main.rs` gains a dedicated `google` arm (it must not fall
through the OpenAI-shaped `_` arm). `Connector::supports_embeddings` -> `true`;
`default_embedding_model` returns a Gemini embedding model.

## Test plan

- `convert_messages`: system -> `systemInstruction`; user/model roles;
  `functionCall` / `functionResponse` round-trip (args/response as JSON objects);
  consecutive-tool-result merge.
- Gemini-specific schema sanitization (mirrors the Bedrock schema tests, extended).
- Streaming parse: text parts, whole-call `functionCall` parts, last
  `usageMetadata` incl. `cachedContentTokenCount`; all usage fields mapped.
- Reasoning: `thinkingBudget` from a Gemini-calibrated budget; omitted when empty.
- Auth: Vertex bearer (refreshing `TokenProvider`, token/SA never logged, never in
  URL) vs Gemini-API `x-goog-api-key` header; host composed from `location`.
- Preflight: missing project / location / credential each yield a specific
  `Unavailable`.
- Error paths (httpmock): 429/`RESOURCE_EXHAUSTED`, quota, `INVALID_ARGUMENT`
  overflow, SAFETY decline, 401, 5xx.
- Embeddings round-trip for both `:predict` and `:embedContent`.
- `MODEL_OVERRIDE` routes the URL path segment.
- Redacting `Debug`; cancellation; malformed-SSE tolerance; callback-abort
  preservation.

## Open questions

- Confirm whether `thinkingConfig` requires `v1beta1` (vs `v1`) on Vertex and pin
  the request version accordingly.
- Vertex token acquisition: reuse the shared OAuth service-account infrastructure,
  or a dedicated JWT-grant via the workspace `jsonwebtoken` (preferred, minimal new
  deps).
- Whether to manage explicit `cachedContents` in a later iteration for large
  reused contexts.
