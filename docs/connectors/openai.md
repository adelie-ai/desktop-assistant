# OpenAI Connector

Crate: `desktop-assistant-llm-openai`

## API Details

- Endpoint: `{base_url}/responses` (Responses API; POST, streaming SSE)
- Embeddings endpoint: `{base_url}/embeddings`
- Auth: `Authorization: Bearer {api_key}`
- Default model: `gpt-5.4`
- Default base URL: `https://api.openai.com/v1`

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Environment | `OPENAI_API_KEY` | Yes |
| Environment | `OPENAI_MODEL` | No |
| Environment | `OPENAI_BASE_URL` | No |
| Config file | `daemon.toml` [openai] section | No |

## Hosted Tool Search

On by default. `OpenAiClient` implements hosted tool search
(`supports_hosted_tool_search()` plus `stream_completion_with_namespaces()`,
which adds a `{"type": "tool_search"}` sentinel to the `tools` array), so an
unconfigured connection gets it. Anthropic defaults the same way.

An endpoint that serves the Responses API but rejects the `tool_search` tool
type does not fail the turn. The client re-sends the same turn with every tool
inline and memoizes the model, so later turns send one request instead of two.
The retry is narrow: it answers a client error that is not authentication, a
timeout, throttling, quota, or a context overflow.

Setting `hosted_tool_search = false` on the connection is still the cheaper
answer for an endpoint known never to serve it, because it also skips the first
refused request. `docs/connectors/cloud-connector-abstraction.md`, section 5,
records the decision and both fallbacks.

## Prompt Caching

OpenAI applies [automatic prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
to all API requests with no opt-in required.  Cached input tokens are billed at
50% of the normal input token price.

### How it works

The API automatically caches the longest matching prefix of the request.  There
are no explicit cache breakpoints or headers to set.  Caching kicks in when the
request prefix is long enough (1024+ tokens as of early 2025).

### Why no code changes are needed

- Caching is fully automatic and server-side
- No request-level annotations (like Anthropic's `cache_control`) exist
- The same dynamic tool list concern applies (tool search changes the tool
  array), but since OpenAI caches the longest matching prefix rather than
  requiring exact prefix matches up to breakpoints, partial cache hits still
  work.  A tool list change only invalidates from the point of divergence
  forward, so the system message and any unchanged tools still get cached

### Dynamic tool list impact

Unlike Anthropic's all-or-nothing breakpoint model, OpenAI's prefix caching is
more forgiving with dynamic tool lists:

- System message prefix: always cached (if long enough)
- Unchanged tool prefix: cached up to the point where the list diverges
- Messages after a tool list change: cache miss only from the divergence point

This means the tool search activation pattern has minimal caching impact on
OpenAI.
