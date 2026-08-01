# Bedrock Connector Design

Crate: `desktop-assistant-llm-bedrock`

## Overview

The Bedrock connector provides unified access to AWS Bedrock with support for multiple backend APIs. It aggregates models across backend APIs while presenting a coherent capability model to clients.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     BedrockConnector                          │
│  - Unified external interface                                 │
│  - High-level concerns: retry, timeouts, cache policy         │
│  - Model aggregation across backends                          │
│  - Capability composition                                      │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    Converse   │   │    Invoke     │   │   Responses   │
│    Backend    │   │    Backend    │   │   Backend     │
│               │   │               │   │  (GPT-5.x)    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                  AWS SDK (bedrock-runtime-client)
```

## Backend APIs

### Converse Backend (Primary)

**Surface:** `Converse` / `ConverseStream` SDK calls

**Use when:** General-purpose model access across Bedrock providers

**Capabilities:**
- Streaming: ✓
- Tool calling: ✓
- Vision: ✓ (model-dependent)
- Tool search: ✗
- Prompt caching: ✗
- Cache control: ✗

**Models:** Most Bedrock models accessible through this API (Anthropic, Amazon Nova, Cohere, etc.)

The Converse API is Bedrock's provider-agnostic abstraction. It normalizes request/response formats across different model providers, which simplifies the connector but loses provider-specific features.

### Invoke Backend (Anthropic Features)

**Surface:** `InvokeModel` / `InvokeModelWithResponseStream` SDK calls

**Use when:** Anthropic-specific features are required

**Capabilities:**
- Streaming: ✓
- Tool calling: ✓
- Vision: ✓
- Tool search: ✓ (Anthropic-style)
- Prompt caching: ✓ (Anthropic-style)
- Cache control: ✓

**Models:** Anthropic models (`anthropic.claude-*`, `us.anthropic.claude-*`)

The Invoke backend sends raw Anthropic-compatible JSON to Bedrock, enabling features that Converse doesn't expose. This duplicates some logic with the direct `llm-anthropic` crate, but the alternative (using that crate directly) would require a different HTTP transport layer.

### Responses Backend (Future)

**Surface:** TBD — AWS Bedrock integration for OpenAI-style Responses API

**Use when:** GPT-5.x models on Bedrock that require the Responses surface

**Capabilities:** (projected)
- Streaming: ✓
- Tool calling: ✓
- Vision: ✓
- Tool search: ✓ (OpenAI-style)
- Reasoning config: ✓ (reasoning_effort)

**Models:** OpenAI models provisioned on Bedrock (if/when AWS launches this)

*Status: Not yet implemented. Placeholder for future Bedrock/Responses integration.*

---

## Backend Trait

```rust
/// Common interface for Bedrock backend APIs.
/// 
/// Each backend (Converse, Invoke, Responses) implements this trait,
/// providing model discovery and completion streaming specific to that API.
#[async_trait]
pub trait BedrockBackend: Send + Sync {
    /// Human-readable API name for logging and diagnostics.
    fn api_name(&self) -> &'static str;
    
    /// List models accessible through this backend.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError>;
    
    /// Stream a completion through this backend.
    async fn stream_completion(
        &self,
        model_id: &str,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        cache_control: Option<CacheControl>,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError>;
    
    /// Capabilities of this backend API.
    fn capabilities(&self) -> BackendApiCapabilities;
    
    /// Whether this backend can serve the given model.
    fn can_serve(&self, model_id: &str) -> bool;
}
```

### Supertrait Relationship

`BedrockBackend` does **not** extend `LlmClient`. Instead:

- `BedrockConnector` (the top-level struct) implements `LlmClient`
- `BedrockConnector` holds `Vec<Arc<dyn BedrockBackend>>`
- The connector delegates to the appropriate backend based on model ID

This separation ensures:
1. Backends focus on API-specific concerns
2. The connector handles cross-cutting concerns (retry, caching, model selection)
3. The `LlmClient` trait остается in `core` and doesn't know about Bedrock internals

---

## Connector Responsibilities

### Model Aggregation

The connector queries all backends for models and merges them:

```rust
async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
    let mut all_models = Vec::new();
    
    for backend in &self.backends {
        let models = backend.list_models().await?;
        all_models.extend(models.into_iter().map(|m| ModelInfo {
            id: m.id,
            display_name: m.display_name,
            context_limit: m.context_limit,
            capabilities: m.capabilities,
            // Annotate with backend info for diagnostics
            backend: Some(backend.api_name().to_string()),
        }));
    }
    
    Ok(all_models)
}
```

Duplicate model IDs (e.g., `anthropic.claude-sonnet-4-6` available via both Converse and Invoke) are resolved by preferring the backend with richer capabilities (Invoke > Converse).

### Backend Selection

When a completion request arrives:

```rust
async fn stream_completion(&self, messages: Vec<Message>, ...) -> Result<LlmResponse, CoreError> {
    let model_id = current_model_override()
        .unwrap_or_else(|| self.default_model.as_str());
    
    // Select backend based on model ID and feature requirements
    let backend = self.select_backend(model_id, /* cache_control, tool_search */);
    
    backend.stream_completion(model_id, messages, ...).await
}
```

Backend selection logic:
1. Check if model ID prefix indicates backend (`invoke/`, `converse/`)
2. Check if request requires features only available on specific backends
3. Default to Invoke for Anthropic models, Converse for others

### Capability Composition

The connector computes effective capabilities for the model picker:

```rust
fn effective_capabilities(&self, model_id: &str) -> EffectiveCapabilities {
    let backend = self.select_backend(model_id);
    let model_caps = self.model_capabilities(model_id);
    let backend_caps = backend.capabilities();
    let connector_caps = self.connector_capabilities();
    
    EffectiveCapabilities {
        can_use_tools: model_caps.tools && backend_caps.supports_tools && connector_caps.supports_tools,
        can_use_vision: model_caps.vision && backend_caps.supports_vision && connector_caps.supports_vision,
        can_use_tool_search: model_caps.tools && backend_caps.supports_tool_search,
        can_use_prompt_caching: backend_caps.supports_cache_control,
        // ... other capabilities
    }
}
```

---

## High-Level Cross-Cutting Concerns

Implemented once in `BedrockConnector`, not per-backend:

### Retry Policy

```rust
// In BedrockConnector::stream_completion
let response = Retryable::new(
    || backend.stream_completion(...),
    |e: &CoreError| matches!(e, CoreError::RateLimited { .. }),
)
.with(backon::ExponentialBuilder::default())
.await?;
```

Retry is configured at the connector level because:
- Retry parameters (max attempts, backoff) are user-facing settings
- All backends share the same transient error patterns (throttling, 5xx)

### Timeouts

- `connect_timeout`: Time to establish connection to Bedrock
- `event_timeout`: Time between streaming events before declaring stall
- `request_timeout`: Overall request timeout (optional, for debugging)

These are connector-level because they're AWS/network-level concerns, not API-level.

### Cache Policy

The connector decides *where* cache breakpoints go; the backend applies them:

```rust
enum CachePolicy {
    /// No caching (Converse backend)
    None,
    
    /// System prompt only (safe default for Invoke)
    SystemPromptOnly,
    
    /// System + selected tools (risky: tool list can change)
    SystemPromptAndTools,
}
```

The connector's `cache_policy` field (from config) drives what `CacheControl` object gets passed to backends. Backends that don't support caching ignore it.

---

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AWS_BEDROCK_API_KEY` | Static credentials (ACCESS_KEY:SECRET[:SESSION_TOKEN]) | — |
| `AWS_PROFILE` | Named AWS profile | — |
| `AWS_REGION` | AWS region | `us-east-1` |
| `BEDROCK_DEFAULT_MODEL` | Default model ID | `us.anthropic.claude-sonnet-4-6` |
| `BEDROCK_CACHE_POLICY` | Cache breakpoint policy | `system_prompt_only` |

### daemon.toml

```toml
[connection.my-bedrock]
type = "bedrock"
region = "us-east-1"
profile = "production"
default_model = "us.anthropic.claude-opus-4-1"
cache_policy = "system_prompt_only"
connect_timeout_secs = 30
event_timeout_secs = 120
```

---

## Model Listing

### Current: Two Parallel Calls

| Call | IAM Action | Contributes |
|------|-----------|-------------|
| `ListFoundationModels` | `bedrock:ListFoundationModels` | Foundation models (on-demand) |
| `ListInferenceProfiles` | `bedrock:ListInferenceProfiles` | Cross-region profiles |

Matches are filtered and merged into a single list.

### extension: Backend-Specific Listings

Each backend contributes its own model list:

```rust
impl BedrockBackend for ConverseBackend {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Call ListFoundationModels + ListInferenceProfiles
        // Filter to models supported by Converse
        // Annotate with Converse capabilities
    }
}

impl BedrockBackend for InvokeBackend {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Call ListFoundationModels for Anthropic models only
        // Annotate with Invoke capabilities (caching, tool search)
    }
}
```

The connector merges and deduplicates.

---

## Capability Examples

### Example 1: Claude Sonnet with Tool Search

Request: Use tool search with `claude-sonnet-4-6`

| Layer | Check | Result |
|-------|-------|--------|
| Model | `tools: true` | ✓ |
| Backend (Converse) | `supports_tool_search: false` | ✗ |
| Backend (Invoke) | `supports_tool_search: true` | ✓ |
| **Decision** | Use Invoke backend | Tool search enabled |

### Example 2: Nova Premier with Vision

Request: Use vision with `amazon.nova-premier`

| Layer | Check | Result |
|-------|-------|--------|
| Model | `vision: true` | ✓ (Nova Premier trained for vision) |
| Backend (Converse) | `supports_vision: true` | ✓ |
| Backend (Invoke) | N/A | Invoke cannot serve Nova |
| **Decision** | Use Converse backend | Vision enabled |

### Example 3: Embedding Model for Chat

Request: Use chat with `amazon.titan-embed-text-v1`

| Layer | Check | Result |
|-------|-------|--------|
| Model | `kind: Embedding` | Embedding model |
| Backend | — | Cannot serve |
| **Decision** | Reject with ToolsUnsupported | Embedding models don't chat |

---

## Migration Path

### Phase 1: Refactor Existing Code

1. Extract Converse-specific code into `ConverseBackend` struct
2. Extract Invoke-specific code into `InvokeBackend` struct (currently stubbed)
3. Keep `BedrockConnector` as the `LlmClient` impl
4. Add `BackendApiCapabilities` to each backend

### Phase 2: Enable Backend Selection

1. Add backend selection logic to connector
2. Default to Converse, use Invoke when features require it
3. Expose backend name in model metadata

### Phase 3: Capability Integration

1. Implement `EffectiveCapabilities` composition
2. Update daemon to query composed capabilities
3. Update clients (GTK, TUI, web) to display capability diagnostics

---

## IAM Requirements

| Permission | Purpose |
|-----------|---------|
| `bedrock:InvokeModel` | Stream completions |
| `bedrock:InvokeModelWithResponseStream` | Stream completions |
| `bedrock:ListFoundationModels` | Model listing |
| `bedrock:ListInferenceProfiles` | Cross-region model listing |

Minimum viable policy for chat-only usage:

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "arn:aws:bedrock:*:::foundation-model/*"
    }
  ]
}
```

For model listing, add:

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
      "Resource": "*"
    }
  ]
}
```

---

## References

- `docs/design/connector-capabilities.md` — Three-layer capability system
- `docs/connectors/cloud-connector-abstraction.md` — Connector uniformity requirements
- `crates/core/src/ports/llm.rs` — `ModelCapabilities`, `LlmClient` trait
- [AWS Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
