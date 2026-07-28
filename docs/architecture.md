# Architecture

The assistant persona is named **Adele**, in reference to the **Adélie penguin**.

## Design Style

The project follows a ports-and-adapters (hexagonal) layout:

- Inbound ports define what the app can do
- Core domain implements behavior without infrastructure coupling
- Outbound ports abstract external systems (LLM, storage, tools)
- Adapters implement protocol/runtime details (D-Bus, OpenAI, Bedrock, MCP)

## Crate Responsibilities

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

## `tui`

- Terminal interface for conversation workflows
- Connects to daemon over D-Bus
- Streams response updates from D-Bus signals

## Runtime Flow (Prompt)

1. TUI (or any D-Bus client) calls `SendPrompt`
2. D-Bus adapter forwards to core service
3. Core requests LLM streaming completion
4. If tool calls are requested, core checks each one against the caller's tool
   allowlist and the turn's provenance gate, then executes the permitted ones
   through the MCP executor
5. D-Bus adapter emits chunk/complete/error signals
6. Client renders updates incrementally

### Tool-provenance gating

A tool result is ordinary context, so instructions hidden in a web page the
model fetched look exactly like instructions the user wrote. `core::tool_provenance`
classifies every shipped tool on two axes - whether an outside party can
influence what it returns, and what it can do - and the turn loop tracks
whether the current turn has taken in externally-controlled bytes.

Once it has, the acting tiers (`mutate`, `network_egress`, `code_execution`,
and anything unclassified) refuse for the rest of that turn. Reading, the
assistant's own note-keeping, and output to the user's own session stay open.
A refusal is a recoverable tool result, so the turn continues and the model can
answer another way; the next turn starts clean.

The change is reported once per turn:

- as `Event::AssistantStatus.capability_change`, a typed
  `TurnCapabilityChange` naming the reason and the closed tiers, for a client
  or an automation
- as the same event's `message`, one line for a person watching

`ToolUsageView.tool_tier` carries the same classification per tool, so an
integrator can tell which tools a conversation uses can be refused mid-turn.

There is no confirmation round-trip. A refused call stays refused for that
turn.
