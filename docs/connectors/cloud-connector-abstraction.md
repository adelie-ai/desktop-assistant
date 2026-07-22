# Cloud Connector Abstraction

Design note shared by the three new connectors: `openrouter`, `google`
(Vertex AI Gemini), and `azure` (Azure OpenAI). It records what every connector
must provide uniformly, what genuinely varies between providers, and where each
varying concern is encapsulated so that `crates/core` never learns a provider's
name.

Reference connectors: `llm-bedrock` (most exercised), `llm-anthropic` (explicit
prompt caching + hosted tool search, redacting `Debug`), `llm-openai` (hosted tool
search, Responses API). Shared helpers: `llm-http`.

Provider-API surfaces were verified against live docs on 2026-07-22; the specific
version strings and field names still carry a "verify at implementation" note
because these APIs rev frequently.

## The uniform contract

Every connector is an `Arc<dyn LlmClient>` (`crates/core/src/ports/llm.rs`). The
daemon wraps it in the same decorator stack regardless of provider
(`ClassifyingLlmClient` innermost, then `RoutingLlmClient` -> `FixedReasoningLlmClient`
-> `RetryingLlmClient` -> `MaybeProfiled`), so a connector only implements the
port and never touches daemon state. Per-turn context (model override, reasoning
config, cancellation, tool allowlist, context budget) arrives through the
task-locals in `core::ports::llm`, not through new arguments.

A connector must implement:

- `stream_completion(messages, tools, reasoning, on_chunk) -> LlmResponse`
- `get_default_model()` / `get_default_base_url()` (static and instance forms;
  the daemon's `Connector::default_*` helpers call the static forms)
- `list_models()` / `refresh_models()` -> curated table merged with the live
  endpoint via `merge_curated_with_live`, behind a TTL cache (see "Shared model
  cache")
- `max_context_tokens()` via `apply_context_cap(context_cap, curated_or_live)`
- Optionally `supports_hosted_tool_search()` + `stream_completion_with_namespaces()`
- Optionally `EmbeddingClient` when the provider serves embeddings

The builder shape is fixed by the factory in `crates/daemon/src/registry.rs`
(`build_llm_client`): `new(api_key)` plus `.with_model / .with_base_url /
.with_temperature / .with_top_p / .with_max_tokens / .with_connect_timeout /
.with_event_timeout / .with_max_context_tokens / .with_hosted_tool_search`, plus
any provider-specific setters. Precedent for provider-specific setters:
`llm-bedrock` adds `.with_aws_profile` / `.with_region`, and its extra state rides
on `ResolvedLlmConfig` (see "Threading provider config"). Azure and Vertex follow
this precedent with their own setters, not by overloading `new(api_key)`.

### Credential redaction is part of the contract

Every connector struct that holds credential material (an API key, an OAuth
access token, or a service-account private key) MUST either not implement `Debug`
or implement a hand-written redacting `Debug`, matching the house standard:
`llm-anthropic/src/lib.rs` renders `api_key` as `<redacted; len=N>`, and
`mcp-client/src/oauth.rs` renders `TokenSet` fields as `<redacted N bytes>`. This
is load-bearing for Vertex, which holds a PEM private key or a live bearer token:
a naive `#[derive(Debug)]`, an error context, or a panic in the stream loop would
otherwise dump the full credential into logs. Credentials travel in headers only,
never in a URL or query string, and never in an error/log message (see the
per-provider error sections).

### No silent failures

Every failure path surfaces something informative and recoverable to the user; a
turn never hangs or dies with an opaque error the user cannot act on. Two
consequences for these connectors:

- **Preflight is auth-aware and field-aware.** `sanity_check_resolved`
  (`registry.rs`) currently only knows "openai/anthropic need a key" and
  "base_url non-empty." That is too coarse for the new providers: Azure with
  Entra needs no api key but does need a resolvable endpoint and a deployment;
  Vertex needs a project, a location, and a resolvable credential (ADC or
  service-account). Each new connector gets a preflight that inspects its typed
  fields and returns a specific `Unavailable { reason }` naming the missing piece,
  rather than passing preflight and failing on the first turn with a raw 401/404.
- **Declines say why.** Business declines (Azure `content_filter`, Gemini
  `SAFETY`) are surfaced with an informative, specific reason the user can
  understand, mapped to a non-error, non-retryable outcome (see "Declines are not
  errors"). "Informative" is the bar: not the raw provider body (which echoes the
  flagged user content), but a clear statement of what was refused and why.

## What is shared, verbatim

These come from `llm-http` and are reused unchanged by all three connectors:

- `send_and_stream` scaffolding: the connect race (`tokio::select!` of `send()`
  against the cancellation token and `STREAM_CONNECT_TIMEOUT`) and the stall loop
  (`next_step` / `StreamStep` against `STREAM_EVENT_TIMEOUT`).
- `parse_retry_after_header`, `build_response`, the `STREAM_*` constants.
- `apply_context_cap`, `merge_curated_with_live` (`llm-http::models`).
- `ToolCallAccumulator<K>` (`core::ports::llm`) for reassembling streamed tool
  calls.
- `bail_for_status` for non-streaming endpoints (models list, embeddings).

The only genuinely provider-specific code in each crate is: the request-body
types plus `convert_messages`, the streaming event enum plus its dispatch, the
tool / namespace serialization, the tool-schema sanitizer, and the
error-classifier free functions. This is the same four-part split the OpenAI and
Anthropic crates already follow.

## The OpenAI-compatible dialect: build in the connector, extract when earned

OpenRouter and Azure both speak the OpenAI **Chat Completions** wire format
(`messages[]`, `tools[].function`, `tool_calls[]`, SSE `choices[].delta`), which
is distinct from `llm-openai` (that crate uses the newer **Responses** API, so it
will not share this code). This dialect does NOT go into `llm-http`: that crate is
chartered as provider-agnostic primitives, and its `streaming.rs` explicitly names
SSE event parsing as the thing that stays in the connector. Dropping a vendor
schema plus a `cache_control` mechanic into it would violate that charter, and the
abstraction is unearned until a second consumer proves the shape.

Plan:

1. Build the Chat Completions dialect (request/response types, `choices[].delta`
   SSE parsing, `tool_calls` accumulation, tool-schema sanitizer, `cache_control`
   helper) as an internal module of `llm-openrouter`.
2. When `llm-azure` becomes the real second consumer, extract that module into a
   dedicated `llm-openai-compat` crate - a distinct wire dialect with its own
   consumer set is the obvious seam a new crate needs. Leave `llm-http` pure
   primitives.

Each connector then supplies only its auth headers, base-URL / endpoint shaping,
caching decision, model addressing, and error classifier.

## Shared model cache

`list_models()` is hit on every model-picker open, and the live listings are
non-trivial (OpenRouter `/models` is a large catalog; Azure/Vertex listings are
also heavy or occasionally unavailable). `llm-bedrock` already solved this with a
`ModelClock` + 1h TTL cache (`llm-bedrock/src/lib.rs`). Lift that pattern into a
small shared helper (in `llm-http` or the compat crate) that all three new
connectors use, and always degrade to the curated table when the live listing
fails, rather than surfacing an error to the picker.

## What varies, and where it is encapsulated

Each row is a concern the request called out ("anything that varies across
providers ... should be abstracted in the provider"). None leak into `core`; each
lives inside its connector crate (or, where a second consumer earns it, inside the
shared compat crate).

### 1. Authentication

| Provider | Mechanism |
|----------|-----------|
| OpenRouter | `Authorization: Bearer <key>` + optional `HTTP-Referer` / `X-Title` attribution headers |
| Azure OpenAI (v1 GA) | `api-key: <key>` header, or `Authorization: Bearer <entra-token>` (Entra ID / managed identity, scope `https://ai.azure.com/.default`) |
| Google Vertex | OAuth2 bearer from a GCP service account / Application Default Credentials, the cloud-credential analogue of Bedrock's AWS chain |

API-key providers read the key at construction. Cloud-credential providers
(Vertex, and Azure-Entra) resolve and refresh a short-lived token via a
`TokenProvider` seam that never logs the token. The Vertex `TokenProvider` reuses
the workspace `jsonwebtoken` crate and the existing OAuth machinery
(`mcp-client::oauth`, keyring `TokenStore`) rather than a vendor SDK, keeping the
new-dependency surface minimal (any new auth crate is gated on a `cve-mcp` /
`cargo audit` scan before first build).

### 2. Model addressing

| Provider | Where the model id lives |
|----------|--------------------------|
| OpenRouter | request body `model: "vendor/model"` (e.g. `anthropic/claude-sonnet-4-6`) |
| Azure OpenAI (v1 GA) | request body `model: "<deployment-name>"`; the deployment is operator-provisioned and its name need not equal the base model |
| Google Vertex | URL path segment (`.../publishers/google/models/{model}:streamGenerateContent`) |

`MODEL_OVERRIDE` (task-local) still selects the logical model per turn; each
connector maps that logical id to the right wire location. For Azure the logical
id is the deployment name; the connector resolves the deployment's base model
(from the live deployments listing when available) to drive reasoning gating,
`context_limit`, and capability flags - not just reasoning.

### 3. Request / response schema

OpenRouter and Azure (v1 GA) speak OpenAI **Chat Completions** via the shared
compat module. Google Vertex uses the Gemini `generateContent` schema
(`contents[].parts[]`, `systemInstruction`, `tools[].functionDeclarations`,
`generationConfig`, `:streamGenerateContent?alt=sse`) - a full custom mapping,
closest in structure to `llm-bedrock`'s Converse mapping.

### 4. Prompt / token caching

The concern the current Bedrock connector handles poorly (Converse exposes no
cache controls). Caching is fully provider-internal: `core` sees only
`TokenUsage::{cache_creation_input_tokens, cache_read_input_tokens}`.

| Provider | Policy |
|----------|--------|
| OpenRouter | unified `cache_control` block markers that OpenRouter translates per routed provider (Anthropic-style `cache_control` <-> OpenAI `prompt_cache_breakpoint` <-> a default 5-minute cache for Anthropic/Google). Mark the system block; OpenRouter normalizes. Read `usage.prompt_tokens_details.cache_write_tokens` -> `cache_creation_input_tokens` and `.cached_tokens` -> `cache_read_input_tokens`. |
| Azure OpenAI | automatic server-side caching; no request annotations. Read `prompt_tokens_details.cached_tokens` -> `cache_read_input_tokens`. |
| Google Vertex | implicit caching on Gemini 2.5+ (no annotation); optional explicit `cachedContents` resources for large reused prefixes (deferred). Read `usageMetadata.cachedContentTokenCount` -> `cache_read_input_tokens`. |

The read side is uniform (fill `TokenUsage`). The write side is one connector's
inline decision (OpenRouter marks the system block; Azure and Vertex-v1 do
nothing). The breakpoint lesson from `anthropic.md` holds: the tool list is
dynamic (runtime tool search mutates it), so the only safe breakpoint is the
system prompt. There is no `PromptCache` type until a second breakpoint-deciding
connector appears; the `cache_control`-into-content-array mechanic lives in the
compat module as a plain helper.

### 5. Dynamic / hosted tool search

Already a trait seam: `supports_hosted_tool_search()` +
`stream_completion_with_namespaces()`. None of the three enable it in v1 -
OpenRouter's routed API does not expose it uniformly, Azure Chat Completions does
not, and Gemini uses function declarations. All three inherit the trait's default
`stream_completion_with_namespaces` (flatten namespaces into `stream_completion`).
The seam is kept so a later version (e.g. Azure on the Responses surface) can opt
in without a core change.

### 6. Reasoning / extended thinking

`ReasoningConfig` (task-local) carries `thinking_budget_tokens` and
`reasoning_effort`. The daemon's `map_effort_to_reasoning_config`
(`api_surface.rs`) decides which field an `Effort` populates per connector, and
the connector translates that field to its wire shape.

| Provider | Field | Daemon arm |
|----------|-------|-----------|
| OpenRouter | `reasoning: { effort }` or `reasoning: { max_tokens }` | own arm (see note) |
| Azure OpenAI | `reasoning_effort` (+ `max_completion_tokens`, not `max_tokens`, for reasoning models) | the `openai` arm (sets `reasoning_effort`) |
| Google Vertex | `generationConfig.thinkingConfig: { thinkingBudget, includeThoughts }` | own `google` arm (sets `thinking_budget_tokens`, Gemini-calibrated) |

Two corrections over the naive plan: (a) Google has no effort literal, so its
daemon arm must feed `thinking_budget_tokens` with Gemini-calibrated budgets, NOT
reuse the Claude `map_anthropic_thinking_budget` table; (b) OpenRouter's
`thinking_budget_tokens -> reasoning.max_tokens` mapping is dead code if OpenRouter
sits in the `openai` arm (which only sets `reasoning_effort`), so OpenRouter needs
its own arm to emit a budget, or the `max_tokens` mapping is dropped. The
daemon-owns-budget-vs-effort / connector-owns-wire-shape split is documented here;
if it proves too tangled, the alternative is to pass the raw `Effort` to
connectors and let each own the full mapping (removing the per-connector arm).

### 7. Error classification and declines

Each connector maps transport failures to `CoreError` variants: `RateLimited`
(retryable, carries `Retry-After`), `QuotaExceeded` (permanent billing),
`ContextOverflow` (with token counts when parseable), `ToolsUnsupported`, else a
clear `Llm` message. This is the sanctioned place for string-matching on provider
error bodies (a documented carve-out kept at the connector boundary); each
provider carries its own classifier free functions with unit tests.

Declines are not errors. A `content_filter` / `SAFETY` outcome is a business
decline, not a technical failure: it is non-retryable, logged at info/debug (never
error), surfaced to the user with a specific, informative reason, and its raw
provider body (which contains the flagged user content) is never dumped into logs
(AGENTS.md 8.2/8.3).

## Threading provider config (do not skip)

The factory builds every client from a single flat `ResolvedLlmConfig`
(`registry.rs`), populated by the resolver (`config/resolution.rs`) and already
carrying provider-specific fields (`aws_profile`, `keep_warm`). Azure needs
`deployment`, `api_surface` (`v1` | `classic`), `auth_mode` (`api_key` | `entra`);
Vertex needs `project`, `location`, `auth_mode` (`vertex` | `api_key`),
`credentials_path`. None exist on `ResolvedLlmConfig` today, and `new(api_key)`
has nowhere to put them. Adding a provider therefore MUST also:

- add the provider-specific fields to `ResolvedLlmConfig` (consider a per-connector
  `extras` sub-struct rather than flattening many optionals onto the shared
  struct), and
- extract them in the resolver's two per-connector matches
  (`config/resolution.rs`), and
- thread them via new builder setters in the factory arm.

A checklist that stops at the config enums leaves Azure/Vertex with no way to
receive their core config.

## Wiring surface (per new first-class provider)

There is no single `ProviderKind` enum; identity is spread across four parallel
enums, four mappers, a string-match factory, and several tables. Some sites are
compile-forced (a missing arm fails the build); several are NOT - a missing arm
compiles and misbehaves silently. Both kinds are listed; the silent ones are
marked, because #4 ("no silent failures") applies to the wiring itself.

Compile-forced (a new variant breaks the build until every arm exists):

- `ConnectionConfigPayload` + `connector_type()` (`core/src/ports/inbound.rs`)
- `ConnectionConfigView` + `connector_type()` (`api-model/src/lib.rs`)
- `ConnectionConfig` (`connector()` + `set_secret()`) and the `Connector` enum
  (`as_str/parse/default_base_url/default_chat_model/default_backend_chat_model/
  default_embedding_model/default_http_base_url`, plus review of
  `supports_embeddings/supports_hosted_tool_search`), plus a `<Provider>Connection`
  struct (`daemon/src/connections.rs`)
- The two resolution matches (`daemon/src/config/resolution.rs`)
- Four mappers (`application/src/lib.rs` x2, `daemon/src/api_surface.rs` x2)
- `model_defaults.rs` `DefaultsFile` field + `defaults_for` arm

Silent if omitted (compiles, misbehaves - each MUST be added):

- `build_llm_client` factory arm (`registry.rs`): the `_` arm builds an
  `OpenAiClient` hitting `/responses`, so an un-armed provider silently posts to
  the wrong endpoint against the provider's host.
- `sanity_check_resolved` (`registry.rs`): the auth-aware preflight from #4; a
  missing entry marks a keyless connection healthy and fails at request time.
- Embeddings factory (`main.rs`): the `_` arm builds an OpenAI-shaped client;
  Google's non-OpenAI embeddings need their own arm, and the availability gate
  (`resolution.rs`, currently the literal `connector != "anthropic"`) must become
  a real `Connector::supports_embeddings` check (which defaults `true`, so
  OpenRouter needs an explicit `false`).
- `map_effort_to_reasoning_config` (`api_surface.rs`): the `_` arm drops reasoning
  silently.
- `Connector::parse` (`connections.rs`): its `_ => None` coerces an unknown
  declared type to `Connector::OpenAi` defaults; landing all enum arms + `parse`
  atomically avoids a declared `type` silently masquerading as OpenAI.
- `connection_from_legacy_llm` (`connections.rs`): legacy `[llm]` path.

Better: where practical, route the factory / sanity / embeddings / reasoning
dispatch through the typed `Connector` enum (issue #47's direction) so a new
variant makes those matches non-exhaustive and the compiler enforces the wiring
instead of leaving it to review.

Env vars: `<CONNECTOR>_API_KEY` / `_MODEL` / `_BASE_URL` are derived from the
connector key (`config/mod.rs`), so the key `azure` derives `AZURE_API_KEY`, not
`AZURE_OPENAI_API_KEY`. Connectors whose conventional env names differ (Azure)
set an explicit default `api_key_env` on their `*Connection` struct rather than
relying on the derived name.

Docs: `docs/connectors/<provider>.md`, an entry in `docs/architecture.md`, and an
onboarding section in `docs/cloud-providers.md` (console URL, credential-acquisition
steps, privacy brief, and a copy-paste minimal `daemon.toml` block - the multi-field
Azure/Vertex configs cannot be expressed with env vars alone).

Tests: a config round-trip test in `connections.rs`, a `registry` build test, and
a negative test that a genuinely unknown `type` still errors. (The existing
`rejects_unknown_type` fixture asserts `type = "gemini"` is rejected; adding a
`google` variant leaves `gemini` unknown, so that fixture keeps passing unchanged -
no edit required.)

## Client configuration surface

Azure (`deployment`, `auth_mode`, `api_surface`) and Vertex (`project`,
`location`, `auth_mode`, `credentials_path`) carry fields the current
`ConnectionConfigView` / `ConnectionConfigPayload` cannot represent, and the KCM /
GTK / TUI connection edit dialogs have no controls for them. v1 ships these two as
**config-file-only** (documented `daemon.toml` blocks in `cloud-providers.md`);
adding the fields to the wire views and the client edit dialogs is a tracked
follow-up. OpenRouter's fields fit the existing OpenAI-shaped view, so it is
GUI-configurable from the start.

## Phased build plan

Each provider is an independently shippable change; the shared wiring files (the
four enums and mappers, `registry.rs`, `model_defaults.toml`) are edited by all
three, so the provider changes are serialized rather than developed in parallel
worktrees.

1. `llm-openrouter` - lowest risk, OpenAI-compatible, unlocks the most models.
   The Chat Completions dialect (types, SSE parsing, tool-schema + empty-key
   sanitizer, `cache_control` helper) starts as an internal module here.
2. `llm-azure` - v1 GA Chat Completions; becomes the second consumer of the
   dialect, at which point it is extracted into `llm-openai-compat`.
3. `llm-google` (Vertex Gemini) - full custom mapping; the largest change.

Robustness the reference connectors proved necessary, carried into all three (per
"if it generalizes, share it; if it applies conceptually but differs, reimplement
using Bedrock as a guide"):

- Tool-schema sanitization (Bedrock strips top-level `oneOf/anyOf/allOf` and
  injects `type`; one bad MCP schema 400s the whole turn). Shared in the compat
  module for OpenRouter/Azure; reimplemented, larger, for Gemini's stricter
  OpenAPI-subset (also drops `$schema`, `additionalProperties`, `$ref`).
- Empty-key tool-input `{"":{}}` sanitization (gpt-oss emits it; OpenRouter routes
  gpt-oss). Shared in the compat module.
- "Streaming with tools unsupported" detection: classify to
  `CoreError::ToolsUnsupported` at minimum; a non-streaming fallback + per-model
  memo is the Bedrock-proven ideal, relevant to OpenRouter's weak-backend long
  tail.
- TTL'd model-list cache (above).

Testing mirrors the reference crates: `httpmock` integration tests with SSE body
fixtures, pure error-classifier unit tests, a cancellation test, tool-accumulator
tests, malformed-SSE tolerance, callback-abort state preservation, and (Anthropic
pattern) a redaction test. Written first (failing), per the repo TDD policy.

## The Google target (decided)

Google = **Vertex AI** (Gemini), the cloud-credential-authenticated,
project/region-scoped gateway that parallels Bedrock. The simpler Gemini API on
`generativelanguage.googleapis.com` (single `GOOGLE_API_KEY`, no project/region)
is folded in as an `auth_mode` variant of the same crate, since both speak the
identical `generateContent` schema and differ only in host and auth. See
`google.md`. Verified current: Vertex `v1` REST
`.../publishers/google/models/{model}:streamGenerateContent` is the live GA
surface (branding is migrating to "Gemini Enterprise Agent Platform" but the REST
path is unchanged); `v1beta1` carries the newest features and may be required for
`thinkingConfig` - confirm at implementation.
