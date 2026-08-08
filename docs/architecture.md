# Architecture

The assistant persona is named **Adele**, in reference to the **Adélie penguin**.

## Design Style

The project follows a ports-and-adapters (hexagonal) layout:

- Inbound ports define what the app can do
- Core domain implements behavior without infrastructure coupling
- Outbound ports abstract external systems (LLM, storage, tools)
- Adapters implement protocol/runtime details (D-Bus, OpenAI, Bedrock, MCP)

## Crate Responsibilities

Every workspace member, in `Cargo.toml` order. The crates with more to say have
their own section below.

| Crate | Responsibility |
|---|---|
| `protocol` | Dependency-light protocol and domain enums, plus the pure rules a domain type and its wire view must answer identically |
| `api-model` | Protocol-neutral command / result / event types shared by every adapter |
| `application` | Maps `api-model` commands onto the core inbound ports |
| `frame-codec` | Length-prefixed frame codec (4-byte LE length + body) shared by the local transports |
| `peer-cred` | Kernel-attested `SO_PEERCRED` identity lookup for a connected Unix socket |
| `auth-jwt` | HS256 JWT claim shape, codec, and key-file IO for the network door |
| `ws-interface` | WebSocket frontend (axum), optional TLS, and the `POST /login` endpoint |
| `core` | Domain entities, inbound and outbound ports, `ConversationHandler` |
| `daemon` | Composition root: wires adapters, serves the transports |
| `llm-openai` | OpenAI Responses / Chat Completions streaming connector |
| `llm-openai-compat` | Building blocks for the OpenAI Chat Completions dialect — a library, not an `LlmClient` |
| `llm-openrouter` | OpenRouter aggregator connector, on the shared compat dialect |
| `llm-azure` | Azure OpenAI (Microsoft Foundry) connector, on the shared compat dialect |
| `llm-bedrock` | Amazon Bedrock connector (Converse, Responses, Invoke) |
| `llm-anthropic` | Anthropic Messages API connector |
| `llm-google` | Google Vertex AI / Gemini `generateContent` connector |
| `llm-ollama` | Ollama connector, implementing both `LlmClient` and `EmbeddingClient` |
| `llm-http` | Shared HTTP error-handling helpers for the reqwest-based connectors |
| `mcp-client` | MCP transport, tool discovery, and per-server tool routing |
| `storage` | Postgres persistence for conversations, knowledge, and assistant state |
| `storage-sqlite` | Embeddable SQLite adapter over the same storage ports, behind an off-by-default `sqlite` feature |
| `client-common` | Shared client-side transport, config, and command types for the frontends |
| `transport-dispatch` | The per-connection request / event loop that WS and UDS both run |
| `uds-interface` | Unix-domain-socket frontend; local clients authenticate by peer-cred |
| `dbus-bridge` | Standalone per-user binary that owns `org.desktopAssistant` |

## `core`

- Domain entities (`Conversation`, `Message`, roles, tool metadata)
- Inbound service traits (`ConversationService`, `AssistantService`)
- Outbound traits (`LlmClient`, `ConversationStore`, `ToolExecutor`)
- `ConversationHandler` orchestration (including tool-call loop)

## `dbus-bridge`

- Standalone per-user binary `adelie-dbus-bridge` that owns `org.desktopAssistant`
- Translates D-Bus method calls into `api::Command`s and ships them to the daemon
  over an authenticated UDS connection — the same hardened path UDS/WS clients use
- Forwards the daemon's signal stream to D-Bus signals
- Replaced the daemon's former in-process `dbus-interface` adapters (cutover #281/#319);
  see [dbus-bridge.md](dbus-bridge.md)

## `daemon`

- Initializes logging, LLM, MCP executor, persistent conversation store
- Wires `ConversationHandler` with adapters
- Serves the local UDS frontend (+ optional WS); no longer claims a session-bus name

## `llm-openai`

- OpenAI-compatible Chat Completions streaming client
- SSE chunk parsing and tool-call delta accumulation
- Converts core messages/tool definitions to provider payloads

## `llm-bedrock`

- Amazon Bedrock ConverseStream API client
- Tool-use mapping between Bedrock content blocks and core tool-call model
- Bedrock `InvokeModel` embedding support for search vectors

## `mcp-client`

- Spawns MCP servers via stdio JSON-RPC
- Discovers tools and routes tool calls per server
- Handles `list_changed` notifications and `listChanged` flags
- Maintains cached tools/resources/prompts metadata
- Runtime enable/disable re-writes the persistent tool-search index
  (`tool_definitions`), not just in-memory state: a `ToolReindexFn` closure
  injected by the daemon (kept storage-free — `mcp-client` never depends on
  `storage`) delete-then-reinserts the `"mcp"` source after each toggle, so a
  hot-enabled server's tools become discoverable — and a hot-disabled server's
  rows are pruned — without a daemon restart. Unwired when there is no Postgres,
  leaving the headless path unchanged.

## Runtime Flow (Prompt)

1. A client calls `SendPrompt` over one of the transports — WebSocket, UDS, or
   D-Bus, which the bridge carries to the daemon over UDS
2. The transport adapter maps the call to an `api::Command`; the shared
   dispatcher hands it to the application layer and on to the core service
3. Core looks the prompt up against memory and surfaces the candidates as a
   `[Recall]` block on the turn's first round - see
   [pre-prompt recall](features/pre-prompt-recall.md). What that block offered,
   and what the model then opened or marked, is recorded in the
   [knowledge use log](features/knowledge-use-log.md), and the situation each
   entry was written in and has proved useful in ranks the candidates against
   the situation this prompt arrived in
4. Core requests LLM streaming completion
5. If tool calls are requested, core checks each one against the caller's tool
   allowlist, the turn's provenance gate, and what this user has been burned by
   before - see [negative memory](features/negative-memory.md) - then executes
   the permitted ones through the MCP executor
6. The dispatcher streams chunk / complete / error events back over the same
   connection; for a D-Bus caller the bridge re-emits them as D-Bus signals
7. Client renders updates incrementally

### Tool-provenance gating

A tool result is ordinary context, so instructions hidden in a web page the
model fetched look exactly like instructions the user wrote. `core::tool_provenance`
classifies every shipped tool on two axes - whether an outside party can
influence what it returns, and what it can do - and the turn loop tracks
whether the current turn has taken in externally-controlled bytes.

Once it has, the acting tiers (`mutate`, `network_egress`, `code_execution`,
and anything unclassified) refuse for the rest of that turn. Two things stay
open: reading, and output to the user's own session. Writing does not, and that
includes the assistant's own memory - a scratchpad note, a pinned note, or a
knowledge-base entry is read back into a later turn, and that turn starts
clean. The loop's own `begin_step` / `complete_step` are intercepted before
dispatch, so planning still works in a tainted turn.

A refusal is a recoverable tool result, so the turn continues and the model can
answer another way. It hands the way forward to the user rather than telling
the model to retry later: the content that may be driving the call is still in
the transcript on the next turn.

Two durable surfaces are written outside the turn loop and are handled at the
write rather than at the gate. Step-planning notes (`begin_step` /
`complete_step`) are intercepted before the gate - the step stack has to close
or the turn's compaction breaks - so the step structure is recorded and the
model's wording is not. A subagent's answer is mirrored onto the session
scratchpad from the completion path; it is kept, because
`get_subagent_status` reads a detached delegation's result from that note and
nowhere else, and stamped, so the two tools that can read it back - that one
and `builtin_scratchpad_search` - close the gate when they do. Pinned notes
render into later turns with no tool in the path and so are not covered; that
needs a durable provenance marker rather than a third special case.

The gate protects the turn that read the content. It does not stop a model that
acts on the same text one turn later - that needs a taint marker persisted on
the ingesting message, which is a later phase.

The change is reported once per turn:

- as `Event::AssistantStatus.capability_change`, a typed
  `TurnCapabilityChange` naming the reason and the closed tiers, for a client
  or an automation
- as the same event's `message`, one line for a person watching

`ToolUsageView.tool_tier` carries the same classification per tool, so an
integrator can tell which tools a conversation uses can be refused mid-turn.

There is no confirmation round-trip. A refused call stays refused for that
turn.
