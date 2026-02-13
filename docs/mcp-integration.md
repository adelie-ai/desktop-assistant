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
