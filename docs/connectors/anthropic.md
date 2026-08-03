# Anthropic Connector

Crate: `desktop-assistant-llm-anthropic`

## API Details

- Endpoint: `{base_url}/v1/messages` (POST, streaming SSE)
- API version header: `2023-06-01`
- Default model: `claude-sonnet-4-6-20260227`
- Default base URL: `https://api.anthropic.com`
- Default max tokens: `8192`

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Environment | `ANTHROPIC_API_KEY` | Yes |
| Environment | `ANTHROPIC_MODEL` | No |
| Environment | `ANTHROPIC_BASE_URL` | No |
| Config file | `daemon.toml` [anthropic] section | No |

## Hosted Tool Search

On by default. `AnthropicClient` implements hosted tool search
(`supports_hosted_tool_search()` plus `stream_completion_with_namespaces()`,
which sends namespace tools as deferred and adds the
`tool_search_tool_regex_20251119` sentinel), so an unconfigured connection gets
it. OpenAI defaults the same way.

Set `hosted_tool_search = false` on the connection to turn it off. Do this for
an endpoint that speaks the Messages API without serving the tool-search beta:
unlike `llm-openai`, this client does not yet re-send the turn with the tools
inline when the endpoint refuses the request, so the turn fails instead.
`docs/connectors/cloud-connector-abstraction.md`, section 5, records the
decision and both fallbacks.

## Prompt Caching

The Anthropic API supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
which reduces cost and latency by caching the prefix of requests that stays
identical across turns. Cached input tokens are 90% cheaper than uncached.

### How prefix caching works

Anthropic caches based on an exact prefix match of the request content.  The
prefix order is fixed: **system prompt -> tools -> messages**.  A `cache_control`
breakpoint marks where the cache boundary sits.  Everything up to the breakpoint
must match exactly for a cache hit; any change in that prefix invalidates the
cache.

The API accepts **at most four `cache_control` breakpoints per request** and
returns a 400 `invalid_request_error` beyond that, so breakpoints are a budget,
not a free annotation.

### What we cache explicitly

**The leading system block** -- and only that one.  `convert_messages` hoists
every `Role::System` domain message into the request's `system` array in order,
then stamps `cache_control: {"type": "ephemeral"}` on the first entry alone.

That first entry is the context assembler's system instruction, static for the
lifetime of a conversation, so it is a cache hit after the first turn.  The
entries behind it are the assembler's per-turn `[..]` blocks -- `[Now]`,
`[Summary of earlier conversation]`, `[Current task]`, `[Working state]`,
`[Plan]`, `[Pinned]`, `[Scratchpad]` -- which it re-surfaces with fresh content
every round (see `crates/core/src/context/mod.rs`, `surfaced_blocks`).  Marking
those is wrong twice over: their prefix differs each turn, so the entry is
written and never read, and a turn that surfaces five or more of them exceeds
the four-breakpoint limit and is rejected outright.

### What we rely on automatic caching for

The Anthropic API also performs automatic caching of long prefixes even without
explicit breakpoints.  We rely on this for the messages portion of the request.

### Why we don't cache the tool list

The tool list is dynamic.  The `builtin_tool_search` core tool allows the LLM to
discover MCP tools at runtime.  When tool search activates new tools, they are
added to the `tools` array passed to subsequent LLM calls (see
`service.rs` `send_prompt()` -- the `activated_tools` HashMap).

Because tools sit between system and messages in the cache prefix order, **any
change to the tool list invalidates the cache for all messages that follow**.
Adding a `cache_control` breakpoint on tools would create cache entries that are
immediately invalidated when the tool list changes, wasting cache write costs.

### Tradeoffs considered and rejected

| Approach | Problem |
|----------|---------|
| Cache breakpoint on last tool | Tool list changes on activation, invalidating the entire message cache that follows |
| Breakpoint on every system block | The assembler surfaces up to eight; past four the API rejects the request with a 400, and the per-turn blocks were never cacheable anyway |
| Truncate at four breakpoints in the connector | Keeps the request legal but still marks volatile blocks, paying cache-write cost for entries that are never read |
| Static core tools + `execute_tool` wrapper | Keeps tools array stable, but loses structured `tool_use` content blocks; adds indirection; the LLM must format calls through a generic wrapper instead of calling tools directly |
| Move activated tools into conversation messages only | Same structured calling loss; also changes the tool discovery contract between service and LLM layers |

### Future considerations

If the tool list stabilizes early in a session (tool search typically fires in
round 1), subsequent rounds benefit from automatic caching of the full
system+tools+messages prefix.  If Anthropic ever supports caching tools
independently from the message prefix, explicit tool caching would become
viable.
