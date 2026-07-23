# Azure OpenAI Connector

Crate: `desktop-assistant-llm-azure` (planned)

Azure OpenAI (Microsoft Foundry) serves models through operator-provisioned
**deployments**. One connector points at one Azure resource and picks a model on
top, exactly like Bedrock - the wrinkle is that the "model" is a deployment you
create in the Azure portal, whose name you choose and which need not equal the
base model. Built on the OpenAI Chat Completions dialect shared with OpenRouter
(`llm-openai-compat`); see `cloud-connector-abstraction.md`.

## API Details

Target the **v1 GA API** (generally available since 2025; the recommended,
non-dated surface):

- Endpoint: `POST {resource}/openai/v1/chat/completions`
- `{resource}`: `https://{name}.openai.azure.com` (also
  `https://{name}.services.ai.azure.com`); the connector appends `/openai/v1`.
  Resource-specific and required - there is no sensible default host.
- Model: the **deployment name** in the request body `model` field.
- **No `api-version` query parameter** (the v1 API removed it).
- Auth: `api-key: {AZURE_OPENAI_API_KEY}` header, or
  `Authorization: Bearer {entra-token}` (Entra ID / managed identity, token scope
  `https://ai.azure.com/.default`) via a refreshing `TokenProvider`.

Legacy fallback (`api_surface = "classic"`): the older
`{resource}/openai/deployments/{deployment}/chat/completions?api-version={ver}`
shape, deployment in the URL path and a dated `api-version`. Still supported by
Azure; offered only for resources not yet on v1. Default is `v1`.

Note: Microsoft recommends the Responses API for first-party Azure models, but v1
Chat Completions is GA, uniform with OpenRouter, and also serves cross-provider
Foundry models (DeepSeek, Grok). Chat Completions is the v1 target here; the
Responses surface is a later option that would also unlock hosted tool search.

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Environment | `AZURE_OPENAI_API_KEY` | Yes (unless Entra) |
| Environment | `AZURE_OPENAI_BASE_URL` (the resource endpoint) | Yes |
| Environment | `AZURE_OPENAI_MODEL` (the deployment name) | Yes |
| Config file | `daemon.toml` `[connections.<id>]` `type = "azure"` | Recommended |

The connector key `azure` would derive `AZURE_API_KEY` from the generic rule, so
`AzureConnection` sets an explicit default `api_key_env = "AZURE_OPENAI_API_KEY"`
(and `AZURE_OPENAI_BASE_URL` / `AZURE_OPENAI_MODEL`) to match Azure's own
conventions. Fields: `deployment` (the body `model`), `api_surface` (`v1` |
`classic`), `auth_mode` (`api_key` | `entra`), `api_version` (classic only),
`base_url`, api_key_env, secret, timeouts, max_context_tokens. `base_url` is the
resource endpoint, so `Connector::default_http_base_url` returns empty; a
missing base_url resolves to a clean `Unavailable` at preflight (see below), not a
default host. The struct holds the api key / token and implements a redacting
`Debug`.

These fields exceed what `ConnectionConfigView` / `ConnectionConfigPayload` can
represent and have no client edit-dialog controls, so **v1 Azure is
config-file-only** with a documented `daemon.toml` block in `cloud-providers.md`;
wiring the fields into the wire views and client dialogs is a tracked follow-up.

## Preflight

Per "no silent failures": an auth-aware, field-aware check that surfaces a
specific reason. Azure requires a non-empty `base_url` and a resolved `model`
(deployment); a key is required only when `auth_mode = api_key`. Missing any of
these returns `Unavailable { reason }` naming the piece (for example "Azure needs
the resource endpoint, e.g. https://<resource>.openai.azure.com"), rather than
passing preflight and returning an opaque 401/404 on the first turn.

## Model addressing and the picker

`MODEL_OVERRIDE` selects the deployment, which the connector puts in the body
`model`. Because deployment names are operator-defined, the connector resolves the
deployment's base model (from the live deployments listing when available) to
drive reasoning gating, `context_limit`, and capability flags uniformly.

The model picker must NOT present curated base-model ids (`gpt-4o`) as selectable
for Azure - picking `gpt-4o` when the deployment is named `my-gpt4` yields a 404.
Prefer the live deployments listing; note that enumeration may be an ARM
control-plane operation (`management.azure.com`) rather than a data-plane path, so
the connector degrades to "enter your deployment name" / curated-labelled-as-base
when the listing is unavailable (the Bedrock swallow-and-log pattern).

## Request / response mapping

OpenAI Chat Completions via the compat module (see `openrouter.md`). Azure-specific
differences: the `/openai/v1` path, the `api-key` header or Entra bearer, and
reasoning models requiring `max_completion_tokens` (not `max_tokens`). Set the
usage-include option so the final chunk carries `usage` with
`prompt_tokens_details.cached_tokens`. Newer reasoning models accept a `developer`
role in place of `system`; the compat module maps `system` appropriately per
model.

## Prompt caching

Automatic server-side caching; no request annotations. Read
`prompt_tokens_details.cached_tokens` -> `cache_read_input_tokens`;
`cache_creation_input_tokens` stays `None` (no explicit write step). The connector
never assumes a hit - it reports only what the API returns.

## Hosted tool search

Off (Chat Completions surface). Namespaces flatten via the default trait method.

## Reasoning

`reasoning_effort` for o-series / GPT-5 reasoning deployments (values are
model-dependent: `none`/`minimal`/`low`/`medium`/`high`/`xhigh`), plus
`max_completion_tokens`. Azure sits in the daemon's `openai` reasoning arm (sets
`reasoning_effort`). Gate per base model resolved from the deployment; when the
base model is unknown, fall back to a name heuristic and omit the field rather than
sending an unsupported one.

## Error mapping and declines

OpenAI-shaped envelope. Classifiers:

- 429 -> `RateLimited` (Azure sends `Retry-After` and `retry-after-ms`; prefer the
  seconds header).
- `context_length_exceeded` -> `ContextOverflow`.
- 401 (bad key / expired Entra token) -> a clear `Llm` message naming the fix,
  never echoing the token.
- invalid/retired `api-version` (classic surface only) -> a specific `Llm` message
  telling the operator to update `api_version`.
- 5xx -> `RateLimited`.
- `content_filter` finish reason / policy violation -> a business decline:
  non-error, non-retryable, surfaced with a specific informative reason, logged at
  info/debug, never dumping the flagged-content body.

## Embeddings

Azure serves `text-embedding-3-*` via deployments. `EmbeddingClient` targets
`{resource}/openai/v1/embeddings` with the embedding deployment as `model`;
`Connector::supports_embeddings` returns `true`. The v1 embeddings shape is
OpenAI-compatible, so the existing embeddings `_` fallthrough in `main.rs` can
serve it with the correct base_url - but the availability gate must be the real
`supports_embeddings` check, not the current `connector != "anthropic"` literal.

## Test plan

- v1 URL shaping (`/openai/v1/chat/completions`, body `model`, no api-version) and
  the classic fallback shape.
- `api-key` header vs Entra bearer (refreshing `TokenProvider`, token never
  logged).
- Preflight: missing base_url / model / (api_key when `auth_mode=api_key`) each
  yield a specific `Unavailable`.
- Message/tool round-trip (shared compat tests + an Azure smoke);
  `max_completion_tokens` for reasoning models.
- `cached_tokens` -> `cache_read_input_tokens`.
- Reasoning gating via deployment -> base-model resolution.
- Error paths (httpmock): 400 overflow, 401, 429 (+retry-after), content-filter
  decline, 5xx.
- Embeddings round-trip via `/openai/v1/embeddings`.
- Redacting `Debug`; cancellation; malformed-SSE tolerance; callback-abort
  preservation.

## Open questions

- Whether to add Entra ID in v1 or ship `api_key` first and add `auth_mode=entra`
  behind the `TokenProvider` seam next (reusing the workspace `jsonwebtoken` +
  existing OAuth machinery; any new Azure identity crate gated on a CVE scan).
- Whether to expose the `classic` surface at all in v1, or v1-only until a real
  legacy resource needs it.
