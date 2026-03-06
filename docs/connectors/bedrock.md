# Bedrock Connector

Crate: `desktop-assistant-llm-bedrock`

## API Details

- API: AWS Bedrock Converse Stream (SDK, not raw HTTP)
- Auth: AWS credential chain (IAM, env vars, profiles, SSO)
- Default model: `anthropic.claude-3-5-sonnet-20241022-v2:0`
- Default region: `us-east-1`

Supports static credentials via `AWS_BEDROCK_API_KEY=ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]`
or the standard AWS SDK credential provider chain.

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Environment | `AWS_BEDROCK_API_KEY` | No (falls back to AWS chain) |
| Environment | Standard AWS env vars | No |
| Config file | `daemon.toml` [bedrock] section | No |

## Prompt Caching

The Bedrock connector uses the
[Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html),
which is a provider-agnostic abstraction over multiple model providers.

### Current status: no explicit caching

The Converse API does not expose Anthropic-style `cache_control` annotations.
While the underlying Anthropic models on Bedrock support prompt caching, the
Converse API's type system (`SystemContentBlock`, `ContentBlock`, etc.) does not
include cache control fields.

### Alternative considered: InvokeModel API

Anthropic prompt caching is available on Bedrock through the raw `InvokeModel`
API, where you send the native Anthropic JSON request format directly.  This
would require:

1. Replacing the Converse API call with `invoke_model_with_response_stream`
2. Building the raw Anthropic request JSON (duplicating logic from `llm-anthropic`)
3. Parsing the raw SSE response (also duplicating `llm-anthropic` logic)
4. Losing the provider-agnostic benefit of the Converse API

This was rejected because:
- It would effectively duplicate the Anthropic connector inside the Bedrock crate
- The Converse API provides value by supporting non-Anthropic models on Bedrock
- The maintenance burden would outweigh the caching savings
- Users who want Anthropic caching can use the direct Anthropic connector instead

## Hosted Tool Search

### Current status: not supported

The Converse API does not support hosted tool search.  Both Anthropic and OpenAI
offer server-side tool search (deferred loading + model-driven discovery), but
the Converse API's `ToolConfiguration` type has no `defer_loading` field or tool
search sentinel equivalent.

Anthropic's tool search *is* available on Bedrock through the raw `InvokeModel`
API (same native JSON format as the direct Anthropic API), but using it has the
same trade-offs as prompt caching — see above.

### Future options

1. **AWS adds tool search to Converse API** — adopt it directly, no refactoring
   needed.

2. **Separate Bedrock Invoke connector** — rather than duplicating serialization
   logic, refactor the existing Anthropic (or OpenAI) connector so its request
   serialization is reusable, then create a thin Bedrock Invoke adapter that
   takes the serialized request and sends it via `invoke_model_with_response_stream`
   instead of the provider's HTTP endpoint.  This would unlock both prompt caching
   and tool search on Bedrock without code duplication.  The Converse-based
   connector would remain for non-Anthropic models.

## Future considerations

If AWS adds cache control or tool search support to the Converse API, we should
adopt it.  The same dynamic tool list constraints documented in `anthropic.md`
would apply — system prompt caching would be the safe choice; tool list caching
would depend on how Bedrock handles prefix invalidation.
