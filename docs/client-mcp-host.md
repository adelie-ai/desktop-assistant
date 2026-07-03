# Client-side MCP host (`client-mcp.toml`)

Adele's brain can run remotely (see the k8s deployment). Tools that must act on
**your** machine — read your files, run a local command — therefore need to
execute on the *edge*, not wherever the brain runs. Each Adele client can host
local MCP servers and expose their tools to the brain as **client-side tools**:
the brain calls the tool, the daemon routes the call back to the connected
client, and the client runs it against the local MCP server.

Which local servers a client exposes is configured here.

## Location

`~/.config/adele/client-mcp.toml` (or `$XDG_CONFIG_HOME/adele/client-mcp.toml`).

This is a **shared, per-machine** file — every Adele client on the box reads it.
That is deliberate: which local tools *exist* is a property of the machine,
while which surface (tui / gtk / voice / kde) *exposes* them is a per-client
choice. Clients on different machines (desktop, a Raspberry Pi, a laptop) each
have their own file, so the same remote brain gets different local reach
depending on which edge is connected.

## Schema

- `[[servers]]` — server *definitions*, mirroring the daemon's `mcp_servers.toml`
  (`name`, `command`, `args`, `namespace`, `enabled`, `env`, `env_secrets`).
  `namespace` exposes a server's tools as `{namespace}__{tool}`, which keeps
  them from colliding with each other or with server-side tools.
- `[surfaces.<name>].enabled` — which defined servers that surface hosts. A
  surface with **no** entry inherits `[surfaces.default]`; an **explicit empty**
  list means "nothing".

Definitions say what exists; surfaces say who gets what.

## Example

```toml
[[servers]]
name = "filesystem"
command = "fileio-mcp"
args = ["serve", "--root", "/home/dave"]
namespace = "fs"

[[servers]]
name = "git"
command = "git-mcp"
namespace = "git"

[[servers]]
name = "browser"
command = "web-mcp"
namespace = "web"

# Applies to any surface without its own entry (e.g. tui).
[surfaces.default]
enabled = ["filesystem", "git"]

# Voice keeps a lean set.
[surfaces.voice]
enabled = ["filesystem"]

[surfaces.gtk]
enabled = ["filesystem", "git", "browser"]
```

## Not to be confused with `mcp_servers.toml`

The daemon's `mcp_servers.toml` configures tools that run **with the daemon**
(in the pod, for a cluster deployment). `client-mcp.toml` configures tools that
run **on the edge**. Same schema, opposite locality.
