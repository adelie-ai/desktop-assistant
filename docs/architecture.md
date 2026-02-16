# Architecture

The assistant persona is named **Adele**, in reference to the **Adélie penguin**.

## Design Style

The project follows a ports-and-adapters (hexagonal) layout:

- Inbound ports define what the app can do
- Core domain implements behavior without infrastructure coupling
- Outbound ports abstract external systems (LLM, storage, tools)
- Adapters implement protocol/runtime details (D-Bus, OpenAI, MCP)

## Crate Responsibilities

## `core`

- Domain entities (`Conversation`, `Message`, roles, tool metadata)
- Inbound service traits (`ConversationService`, `AssistantService`)
- Outbound traits (`LlmClient`, `ConversationStore`, `ToolExecutor`)
- `ConversationHandler` orchestration (including tool-call loop)

## `dbus-interface`

- Exposes `org.desktopAssistant.Conversations`
- Maps D-Bus methods to `ConversationService`
- Emits streaming signals for chunk/complete/error events

## `daemon`

- Initializes logging, LLM, MCP executor, in-memory store
- Wires `ConversationHandler` with adapters
- Registers service on session bus name `org.desktopAssistant`

## `llm-openai`

- OpenAI-compatible Chat Completions streaming client
- SSE chunk parsing and tool-call delta accumulation
- Converts core messages/tool definitions to provider payloads

## `mcp-client`

- Spawns MCP servers via stdio JSON-RPC
- Discovers tools and routes tool calls per server
- Handles `list_changed` notifications and `listChanged` flags
- Maintains cached tools/resources/prompts metadata

## `tui`

- Terminal interface for conversation workflows
- Connects to daemon over D-Bus
- Streams response updates from D-Bus signals

## Runtime Flow (Prompt)

1. TUI (or any D-Bus client) calls `SendPrompt`
2. D-Bus adapter forwards to core service
3. Core requests LLM streaming completion
4. If tool calls are requested, core executes tools through MCP executor
5. D-Bus adapter emits chunk/complete/error signals
6. Client renders updates incrementally
