# Adding MCP Services

## Why MCP Services Matter

The Adelie platform uses [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/) as its primary mechanism for giving the LLM access to tools — file I/O, web search, calendar access, system control, and so on. **Without at least one MCP server configured, the assistant has very limited ability to take actions on your behalf.**

The built-in tools (preference memory, factual memory) are always available, but real-world usefulness depends heavily on external MCP servers providing capabilities relevant to your workflow.

These MCP servers are not the full extent of Adelie platform's functionality. It is usually capable of working out very complex tasks for which it has not been explicitly programmed. You should think of these MCP servers as the building blocks it uses to synthesize more complex behaviors. By providing deterministic abstractions over complex behaviors in this way, the service doesn't need to think so hard and can worry about other things and get to your end result more quickly. 

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

| Field       | Required | Description                                                                          |
|-------------|----------|--------------------------------------------------------------------------------------|
| `name`      | yes      | Logical label for this server; used in logs and startup diagnostics                  |
| `command`   | for stdio | Executable to spawn — must be on `$PATH` or an absolute path. Omit when using `[servers.http]` |
| `args`      | no       | Command-line arguments passed to the process (default: empty list)                   |
| `namespace` | no       | If set, all tools from this server are exposed as `{namespace}__{tool_name}`; if absent, tool names are passed through unchanged |
| `[servers.http]` | no  | Reach the server over HTTP instead of spawning `command` — see [Remote (HTTP) MCP Servers](#remote-http-mcp-servers) |

The daemon communicates with each server over stdio using the MCP JSON-RPC protocol.

## Tool Namespacing

By default, tool names are passed through exactly as the MCP server reports them. Set the optional `namespace` field to prefix all tools from that server:

```
{namespace}__{tool_name}
```

For example:

```toml
[[servers]]
name      = "fileio"
command   = "fileio-mcp"
args      = ["serve", "--mode", "stdio"]
namespace = "fs"
```

This exposes `fileio-mcp`'s `fileio_read_file` as `fs__fileio_read_file`.

**When to use namespacing:**

- **Collision avoidance** — multiple servers that expose tools with the same name (for example, `open_ticket` from a built-in tasks server, Jira, and Bugzilla):

```toml
[[servers]]
name      = "tasks-builtin"
command   = "tasks-mcp"
namespace = "tasks"

[[servers]]
name      = "jira"
command   = "jira-mcp"
namespace = "jira"

[[servers]]
name      = "bugzilla"
command   = "bugzilla-mcp"
namespace = "bz"
```

This exposes `tasks__open_ticket`, `jira__open_ticket`, and `bz__open_ticket` as distinct tools.

- **Multiple instances of the same server** — two `fileio-mcp` processes scoped to different directories:

```toml
[[servers]]
name      = "work-files"
command   = "fileio-mcp"
args      = ["--root", "/home/user/work"]
namespace = "work"

[[servers]]
name      = "personal-files"
command   = "fileio-mcp"
args      = ["--root", "/home/user/personal"]
namespace = "personal"
```

This exposes `work__fileio_read_file` and `personal__fileio_read_file` as distinct tools.

When `namespace` is absent, tool names are forwarded to the LLM exactly as reported by the server — suitable for servers that already use unique, self-describing names (`fileio_read_file`, `terminal_execute`, etc.).

## Multiple Servers

Add as many `[[servers]]` blocks as needed:

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

## Remote (HTTP) MCP Servers

Besides spawning a local process over stdio, the daemon can reach a **remote** MCP server over HTTP (the MCP *streamable-HTTP* transport). Add a `[servers.http]` table instead of a `command`:

```toml
[[servers]]
name      = "gmail-personal"
namespace = "gmail_personal"

[servers.http]
url                = "https://gmailmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_personal_token"
```

Fields under `[servers.http]`:

| Field                | Required | Description                                                                                |
|----------------------|----------|--------------------------------------------------------------------------------------------|
| `url`                | yes      | Remote MCP endpoint (`http://` or `https://`). Its presence selects the HTTP transport      |
| `auth_bearer_secret` | no       | Secret **ID** (looked up in `secrets.toml`) whose value is sent as `Authorization: Bearer`  |

The bearer token itself is never written in `mcp_servers.toml` — only the secret **ID** is. Put the real token in `secrets.toml` (also enforced `0600`):

```toml
# ~/.config/desktop-assistant/secrets.toml
[secrets]
google_personal_token = "ya29.a0Af..."
```

> **Token acquisition is out of scope for the daemon.** Whatever value you place in `secrets.toml` is sent verbatim as the bearer token; obtaining and refreshing it is currently your responsibility (e.g. an OAuth 2.0 access token from your own Google OAuth client).

### Google Workspace (Gmail / Calendar / Drive / Chat)

Google hosts a first-party MCP endpoint per Workspace service; each is one `[[servers]]` entry:

```toml
[[servers]]
name = "gmail"
namespace = "gmail"
[servers.http]
url = "https://gmailmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_token"

[[servers]]
name = "calendar"
namespace = "calendar"
[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_token"
```

(Which tools — and whether writes like sending mail or RSVPing invites are permitted — depends on the OAuth scopes granted to your token.)

**Multiple accounts.** Give each account its own entry with a distinct `namespace` and `auth_bearer_secret`, so the assistant can tell them apart ("create an invite on my *work* calendar" → the `calendar_work__` tools):

```toml
[[servers]]
name = "calendar-personal"
namespace = "calendar_personal"
[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_personal_token"

[[servers]]
name = "calendar-work"
namespace = "calendar_work"
[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_work_token"
```

Within a single account, choosing between that account's calendars (primary vs. a shared "XYZ" calendar) is handled by the server's own `calendarId` tool argument, not by configuration.

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
