# MCP Integration

## Configuration

Daemon loads MCP server config from:

- `$XDG_CONFIG_HOME/desktop-assistant/mcp_servers.toml`
- fallback: `~/.config/desktop-assistant/mcp_servers.toml`

Format:

```toml
[[servers]]
name = "fileio"
command = "fileio-mcp"
args = ["serve", "--mode", "stdio"]
```

## Startup Behavior

- Daemon starts each configured MCP process
- Executor loads initial:
  - tools (`tools/list`)
  - resources (`resources/list`, if implemented)
  - prompts (`prompts/list`, if implemented)

If `resources/list` or `prompts/list` are unsupported (`-32601`), startup continues.

## Dynamic List Refresh

Client handles both MCP patterns:

- notifications:
  - `notifications/tools/list_changed`
  - `notifications/resources/list_changed`
  - `notifications/prompts/list_changed`
- response result flag:
  - `listChanged: true`

When a list is marked changed, executor refreshes the affected cache before serving metadata or executing tools.

## Tool Routing

- Tools are mapped by name to server index
- `execute_tool(name, args)` resolves server via routing table
- Calls are forwarded as `tools/call`

## What tool discovery tells the model

`builtin_tool_search` reports where each hit runs, because a tool's name and
description do not say which machine it acts on. Each result carries a
`runs_on` value:

| `runs_on` | What it means |
|---|---|
| `daemon` | A built-in, or an MCP server the daemon spawned. Acts on the daemon's own files and processes. |
| `remote-service` | An MCP server the daemon reaches over HTTP. Acts on that service, and on no local files. |
| `device` | A tool the connected client registered. Acts on the user's own machine. |

The daemon and remote-service split is read live from the routing table and the
server configuration, so a server added since startup classifies correctly. A
name the executor does not route is a built-in, which runs inside the daemon
process, so it reports `daemon`.

Client-registered tools are searched too. They are registered per connection and
never written to the tool registry, so a search that consulted only the registry
could never offer the option that acts on the user's own machine. They are
matched lexically against the query - the set is tens of tools, and no embedding
exists for it.

Each response also carries `same_machine` and a one-line `runs_on` legend naming
the daemon's machine. Only the runner values present in the results are
described. When the daemon and the client are the same machine, a client tool
and a daemon tool of the same name are the same capability, so the daemon entry
is kept and the duplicate is dropped - matching how the turn loop resolves that
collision.

A search that matched more client tools than it returned reports the count it
dropped in `more_device_tools_matched`.

## Current Surface

Exposed by `McpToolExecutor`:

- `available_tools()`
- `available_resources()`
- `available_prompts()`
- `execute_tool(name, arguments)`

## Test Coverage

- Unit tests for parsing and list-change detection
- Real-server e2e (`fileio-mcp`) for tool flow
- Dynamic mock e2e for live `list_changed` cache refresh
- Spawned-child environment isolation (the pass-through allowlist described in
  [mcp-services.md](mcp-services.md#environment-variables)) and exit-status
  diagnostics for a server that fails to start (`tests/env_isolation.rs`)
