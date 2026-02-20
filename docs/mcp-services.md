# Adding MCP Services

## Why MCP Services Matter

The Adelie platform uses [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/) as its primary mechanism for giving the LLM access to tools — file I/O, web search, calendar access, system control, and so on. **Without at least one MCP server configured, the assistant has very limited ability to take actions on your behalf.**

The built-in tools (preference memory, factual memory) are always available, but real-world usefulness depends heavily on external MCP servers providing capabilities relevant to your workflow.

## Available MCP Servers

The following MCP servers (not an exhaustive list) are developed alongside the Adelie platform and are designed to work with it out of the box.

### fileio-mcp

File system operations for LLM agents: read, write, structured edit, line-aware read, file/content search, copy, move, stat, mkdir, remove, symlinks, permissions, and more.

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]
```

### terminal-mcp

Shell execution for LLM agents. Exposes `terminal_execute` plus a dynamic script tool lifecycle (`terminal_store_script`, `terminal_remove_script`, `terminal_list_scripts`, and per-script `script_<name>` tools). Results include `exit_code`, `stdout`/`stderr`, timeout status, and truncation flags.

```toml
[[servers]]
name    = "terminal"
command = "terminal-mcp"
args    = ["serve", "--mode", "stdio"]
```

> **Note:** Terminal execution is a high-privilege capability. Audit logging is available via `MCP_TERMINAL_LOG_DIR`.

### tasks-mcp

Local task management backed by Markdown files with YAML frontmatter. Supports multiple lists/contexts, a simple `epic → deliverable` hierarchy, and atomic file writes. Storage defaults to `~/.local/share/desktop-assistant/tasks/`.

```toml
[[servers]]
name    = "tasks"
command = "tasks-mcp"
args    = ["serve", "--mode", "stdio"]
```

### timeclock-mcp

Time tracking for projects. Tools: `timeclock_project_upsert`, `timeclock_project_list`, `timeclock_project_delete`, `timeclock_clock_in`, `timeclock_clock_out`, `timeclock_session_get_active`, `timeclock_session_query`.

```toml
[[servers]]
name    = "timeclock"
command = "timeclock-mcp"
args    = ["serve", "--mode", "stdio"]
```


## Configuration File

MCP servers are configured in:

```
$XDG_CONFIG_HOME/desktop-assistant/mcp_servers.toml
```

Which typically resolves to:

```
~/.config/desktop-assistant/mcp_servers.toml
```

Create this file if it does not exist. The daemon reads it at startup; restart the daemon after any changes.

## File Format

Each MCP server is declared as a `[[servers]]` entry:

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]
```

Fields:

| Field     | Required | Description                                                        |
|-----------|----------|--------------------------------------------------------------------|
| `name`    | yes      | Logical name for this server; used in logs and tool routing        |
| `command` | yes      | Executable to spawn — must be on `$PATH` or an absolute path       |
| `args`    | no       | Command-line arguments passed to the process (default: empty list) |

The daemon communicates with each server over stdio using the MCP JSON-RPC protocol.

## Multiple Servers

Add as many `[[servers]]` blocks as needed. Tool names must be unique across all servers; if two servers expose a tool with the same name, the first one registered wins.

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]

[[servers]]
name    = "websearch"
command = "websearch-mcp"
args    = []

[[servers]]
name    = "calendar"
command = "/opt/my-mcp-servers/calendar-mcp"
args    = ["--profile", "work"]
```

## Startup Behaviour

When the daemon starts:

1. Each configured server process is spawned.
2. The daemon performs the MCP `initialize` handshake.
3. `tools/list`, `resources/list`, and `prompts/list` are fetched from each server.
4. A routing table is built mapping tool names → server index.

If a server fails to start, a warning is logged and the daemon continues without that server's tools. No server failure is fatal to the daemon.

## Verifying Loaded Tools

Check daemon logs to confirm servers and tools were loaded:

```bash
just backend-logs
# or, for the dev daemon:
just backend-dev-logs
```

Look for lines like:

```
INFO connecting to MCP server 'fileio': fileio-mcp
INFO MCP server 'fileio' provides 8 tools
```

If a server failed to connect you will see:

```
ERROR failed to connect to MCP server 'fileio': ...
```

## Applying Changes

After editing `mcp_servers.toml`, restart the daemon to reload:

```bash
just backend-restart
# or for the dev daemon:
just backend-dev-restart
```

## Further Reading

- [MCP Integration internals](mcp-integration.md) — protocol details, dynamic list refresh, tool routing
- [D-Bus API](dbus-api.md) — how clients invoke tools via the conversation API
