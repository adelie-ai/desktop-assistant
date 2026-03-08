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

### What we cache explicitly

**System prompt** -- The system prompt is sent as a structured block with
`cache_control: {"type": "ephemeral"}`.  It is static for the lifetime of a
conversation, so this is always a cache hit after the first turn.

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
| Static core tools + `execute_tool` wrapper | Keeps tools array stable, but loses structured `tool_use` content blocks; adds indirection; the LLM must format calls through a generic wrapper instead of calling tools directly |
| Move activated tools into conversation messages only | Same structured calling loss; also changes the tool discovery contract between service and LLM layers |

### Future considerations

If the tool list stabilizes early in a session (tool search typically fires in
round 1), subsequent rounds benefit from automatic caching of the full
system+tools+messages prefix.  If Anthropic ever supports caching tools
independently from the message prefix, explicit tool caching would become
viable.
