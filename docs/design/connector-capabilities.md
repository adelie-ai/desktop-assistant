# Connector Capabilities: A Three-Layer System

## Overview

Every LLM connector in desktop-assistant exposes capabilities at three distinct layers. This document describes the universal pattern and the filtering principle that composes capabilities across layers.

## The Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     CONNECTOR LAYER                         │
│  "Does Bedrock (vs OpenAI vs Anthropic) support X?"         │
│  e.g., AWS credential chain, Bedrock API quirks             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND API LAYER                         │
│  "Does the Converse/Invoke/Responses API support X?"        │
│  e.g., streaming, tool search, cache controls               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                            │
│  "Is this specific model trained for X?"                    │
│  e.g., vision, reasoning, tools, embedding                   │
│  (already exists as ModelCapabilities)                       │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Connector Capabilities

**Question:** "Does this connector implementation support X?"

This layer captures provider-level concerns that apply regardless of which backend API or model is used:

- **Credential handling**: AWS credential chain vs static API key vs OAuth
- **Region/endpoint selection**: `us-east-1` vs `eu-west-1`, etc.
- **Connector-wide limits**: Bedrock's different throughput modes
- **Cross-cutting concerns**: Retry policy, timeout defaults, logging shape

Examples:
- `supports_aws_profiles: bool` — can the connector use named AWS profiles?
- `supports_cross_region_inference: bool` — does the connector know about inference profiles?

### Layer 2: Backend API Capabilities

**Question:** "Does this specific API surface support X?"

Most providers expose multiple API surfaces with different capabilities. This layer captures what a *particular API* can do:

| Backend API | Example Capabilities |
|-------------|---------------------|
| Converse | `supports_tool_search: false`, `supports_streaming: true`, `supports_cache_control: false` |
| Invoke | `supports_tool_search: true`, `supports_streaming: true`, `supports_cache_control: true` |
| Responses (OpenAI) | `supports_tool_search: true`, `supports_streaming: true`, `supports_reasoning_effort: true` |
| Chat Completions (OpenAI) | `supports_tool_search: false`, `supports_streaming: true`, `supports_cache_control: false` |

This is where API-level constraints live:
- Anthropic's Messages API supports tool search; the Converse API does not
- OpenAI's Responses API supports reasoning effort; Chat Completions does not
- Invoke supports prompt caching; Converse does not

### Layer 3: Model Capabilities

**Question:** "Is this specific model trained for X?"

This is the existing `ModelCapabilities` struct:

```rust
pub struct ModelCapabilities {
    pub reasoning: bool,   // Extended-thinking / reasoning traces
    pub vision: bool,      // Image input
    pub tools: bool,       // Tool/function calling
    pub kind: ModelKind,   // Generative vs Embedding
}
```

This layer captures model-level training decisions. A model that can't see images fails vision regardless of what the connector or backend API support.

---

## The Filtering Principle

**Core rule:** Each layer can *block* a capability, never *grant* it independently.

The effective capability is the logical AND across all layers:

```
effective_capability = model.can_X && backend_api.can_X && connector.can_X
```

This composition happens where the layers meet: typically in the daemon's routing layer or when constructing a view for the client.

### Example: Can we use hosted tool search?

| Layer | Capability | Value | Why |
|-------|-----------|-------|-----|
| Model | `tools: bool` | `true` | Claude 4.x trained for tools |
| Backend API | `supports_tool_search: bool` | `false` | Converse API lacks tool search |
| Connector | (delegates to backend) | — | Bedrock connector uses Converse by default |
| **Effective** | `can_use_tool_search` | **`false`** | Backend API blocks |

If we switch to the Invoke backend:
| Layer | Capability | Value | Why |
|-------|-----------|-------|-----|
| Model | `tools: bool` | `true` | Claude 4.x trained for tools |
| Backend API | `supports_tool_search: bool` | `true` | Invoke API supports Anthropic's tool search |
| Connector | (delegates to backend) | — | Bedrock connector using Invoke |
| **Effective** | `can_use_tool_search` | **`true`** | All layers pass |

### Example: Can we use vision?

| Layer | Capability | Value | Why |
|-------|-----------|-------|-----|
| Model | `vision: bool` | `false` | Embedding model, not trained for images |
| Backend API | `supports_vision: bool` | `true` | Converse supports image blocks |
| Connector | `supports_vision: bool` | `true` | Bedrock connector handles images |
| **Effective** | `can_use_vision` | **`false`** | Model blocks |

The model's training decision overrides everything downstream.

---

## Rust Types

### ConnectorCapabilities (Connector Layer)

```rust
/// Capabilities that apply to an entire connector implementation.
/// 
/// These are provider-level concerns that hold regardless of which
/// backend API or model is selected.
#[derive(Debug, Clone, Default)]
pub struct ConnectorCapabilities {
    /// Can use AWS named profiles for auth (Bedrock-specific).
    pub supports_aws_profiles: bool,
    
    /// Can route to cross-region inference profiles.
    pub supports_cross_region_inference: bool,
    
    /// Connector can serve embedding requests.
    pub supports_embeddings: bool,
    
    /// Connector has a live model-listing endpoint.
    pub supports_model_listing: bool,
}
```

### BackendApiCapabilities (Backend API Layer)

```rust
/// Capabilities of a specific backend API surface.
/// 
/// Most providers expose multiple APIs with different capabilities.
/// This captures what a particular API can do, independent of model.
#[derive(Debug, Clone, Default)]
pub struct BackendApiCapabilities {
    /// API supports streaming responses.
    pub supports_streaming: bool,
    
    /// API supports server-side tool search / deferred loading.
    pub supports_tool_search: bool,
    
    /// API accepts cache_control annotations.
    pub supports_cache_control: bool,
    
    /// API supports reasoning/thinking configuration.
    pub supports_reasoning_config: bool,
    
    /// API supports image/vision input.
    pub supports_vision: bool,
    
    /// API supports tool/function calling.
    pub supports_tools: bool,
}
```

### ModelCapabilities (Model Layer)

Already exists in `core/src/ports/llm.rs`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub reasoning: bool,
    pub vision: bool,
    pub tools: bool,
    pub kind: ModelKind,
}
```

---

## Composition in Practice

### For Clients (GUI)

The daemon computes effective capabilities and exposes them via the model-listing endpoint:

```rust
pub struct EffectiveCapabilities {
    // Composed from all three layers
    pub can_use_tools: bool,
    pub can_use_vision: bool,
    pub can_use_reasoning: bool,
    pub can_use_tool_search: bool,
    pub can_use_prompt_caching: bool,
    
    // Layer details for diagnostics
    pub model: ModelCapabilities,
    pub backend_api: BackendApiCapabilities,
    pub connector: ConnectorCapabilities,
}
```

The client sees a single boolean per capability, computed by the daemon. The layer breakdown is available for tooltips/debugging but the primary UI surface is the composed result.

### For Routing (Daemon)

When the daemon decides whether to use hosted tool search:

```rust
fn can_use_hosted_tool_search(
    model_caps: &ModelCapabilities,
    backend_caps: &BackendApiCapabilities,
) -> bool {
    // Model must support tools, API must support tool search
    model_caps.tools && backend_caps.supports_tool_search
}
```

### For Connectors

Connectors expose multiple capability sets:

```rust
trait LlmClient {
    /// Model-level capabilities (existing)
    fn model_capabilities(&self, model_id: &str) -> Option<ModelCapabilities>;
    
    /// Backend API capabilities for the active backend
    fn backend_api_capabilities(&self) -> BackendApiCapabilities;
    
    /// Connector-level capabilities
    fn connector_capabilities(&self) -> ConnectorCapabilities;
}
```

---

## Why Three Layers?

The three-layer design reflects real constraints in the LLM ecosystem:

1. **Backend diversity is real.** A single connector may use multiple APIs with radically different capabilities. Bedrock alone has Converse (no caching, no tool search) and Invoke (caching, tool search). OpenAI has Chat Completions (no reasoning effort) and Responses (reasoning effort). Pretending all backends are the same loses information.

2. **Composition is safer than inheritance.** A capability tree keyed by `(connector, backend_api, model)` is simpler than trying to build a hierarchy. Each layer filters independently.

3. **The filtering principle is universal.** "Each layer can block, never grant" applies everywhere. A model that can't see images won't gain vision from the API. An API that can't cache won't gain caching from the model. This invariant simplifies reasoning.

4. **Diagnostics need layer visibility.** When a user asks "why can't I use tool search?", the answer must name the blocking layer. Returning a single `false` is correct but useless for troubleshooting.

---

## Implementation Notes

### Defaults

Each layer provides sensible defaults:

- `ConnectorCapabilities::default()` — all `false` except what the connector explicitly overrides
- `BackendApiCapabilities::default()` — all `true` for APIs that haven't opted into explicit capabilities (progressive enhancement)
- `ModelCapabilities::default()` — all `false`, `kind: Unknown` (as currently implemented)

The filtering principle means `false` at any layer wins. A model that hasn't been classified blocks capabilities safely.

### Evolution

New capabilities are added to the layer where they belong:

- `supports_parallel_tool_calls` → Backend API layer (API feature)
- `supports_system_prompt_caching` → Backend API layer (API feature)
- `supports_code_interpretation` → Model layer (model training)

Avoid the temptation to flatten capabilities into a single struct. The layer distinction carries semantic information that helps future maintainers understand *why* a capability lives where it does.

### Migration

Existing connectors can adopt incrementally:

1. Add `connector_capabilities()` with hardcoded values for known connector-level features
2. Add `backend_api_capabilities()` with hardcoded values for the primary API
3. Multi-backend connectors (like Bedrock) return different backend capabilities per selected API
4. The daemon's composition logic picks up the new layers automatically

No breaking changes to the trait — the new methods have default implementations returning the progressive-enhancement defaults.

---

## Open Questions

1. **Should backend capabilities be per-model or per-API?** Currently designed as per-API, but some providers have model-level restrictions within an API (e.g., only some models support vision in Chat Completions). If this becomes common, we may need `ModelCapabilities` to capture more.

2. **Where does capability composition happen?** Current design: daemon computes effective capabilities for the client. Alternative: each connector computes its own effective capabilities. The daemon approach keeps connectors simple but centralizes logic; the connector approach is more flexible but duplicates the filtering rule.

3. **How do dynamic backends report capabilities?** The proposed unified Bedrock connector selects a backend at runtime. Backend capabilities would be determined by which backend was selected. This suggests backend capabilities are a *query*, not a *static property*.

---

## References

- `crates/core/src/ports/llm.rs` — `ModelCapabilities`, `LlmClient` trait
- `docs/connectors/bedrock.md` — Bedrock connector design (will be updated)
- `docs/connectors/cloud-connector-abstraction.md` — Connector uniformity requirements
