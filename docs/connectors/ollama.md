# Ollama Connector

Crate: `desktop-assistant-llm-ollama`

## API Details

- Endpoint: `{base_url}/api/chat` (POST, NDJSON streaming)
- Auth: none (local service)
- Default model: `llama3.2`
- Default base URL: `http://localhost:11434`

The connector auto-pulls missing models on first use via `POST /api/pull`.

## Configuration

| Source | Variable | Required |
|--------|----------|----------|
| Config file | `daemon.toml` [ollama] section | Yes (model + base_url) |

## Prompt Caching

Ollama runs inference locally.  There is no API-level prompt caching concept and
no billing, so caching for cost reduction is not applicable.

### KV cache reuse

Ollama's inference server reuses the KV cache for the prompt prefix
automatically at the inference engine level (llama.cpp).  If consecutive
requests share a common prefix, the server skips re-evaluating those tokens.
This is transparent to the client and requires no request-level annotations.

### Why no code changes are needed

- No cost savings possible (local inference, no per-token billing)
- KV cache reuse is handled entirely server-side
- No API surface for cache control exists
