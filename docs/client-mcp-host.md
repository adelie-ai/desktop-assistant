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
- `[surfaces.<name>].disabled_builtins` — built-in (compiled-in) server names this
  surface has turned **off** (see [Built-in servers](#built-in-in-process-servers)).
  Unlike `enabled`, this list is per-surface with **no** `default` fallback and is
  omitted when empty, so a file written before this key existed leaves every
  built-in on.

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

# Voice keeps a lean set, and turns the built-in 'web' server off.
[surfaces.voice]
enabled = ["filesystem"]
disabled_builtins = ["web"]

[surfaces.gtk]
enabled = ["filesystem", "git", "browser"]
```

## Built-in (in-process) servers

Desktop clients ship with a small set of MCP servers **compiled into the binary**
and hosted **in-process** — no subprocess, no `[[servers]]` entry, useful out of
the box on a fresh install. The default desktop set is `fileio`, `terminal`,
`tasks`, and `web`; each client chooses its own compiled-in set at build time
(voice, for example, compiles them in but defaults them all off). A built-in's
tools are namespaced, registered, and routed exactly like a subprocess server's —
the brain cannot tell the two apart.

Each built-in is in one of three states, in precedence order:

1. **Overridden** — a `[[servers]]` entry with the **same name** shadows the
   built-in. The configured (external) server hosts those tools instead; the
   built-in is not run. Use this to swap in a newer or patched external binary
   without losing the tool. The panel shows the built-in as a disabled row
   ("overridden by the external ...").
2. **Disabled** — the built-in's name is listed in this surface's
   `disabled_builtins`. It is not hosted at all, for that surface only. This is
   the explicit off-switch; it takes display precedence over an override. Toggle
   it from a client's MCP panel, or from the CLI:

   ```sh
   adele config mcp disable web        # turn the built-in 'web' off for this surface
   adele config mcp enable web         # turn it back on
   ```
3. **Active** (default) — compiled in, not overridden, not disabled: hosted
   in-process.

Disable is **per-surface**: a hands-free surface (voice) can silence a built-in
the desktop keeps. Because the in-process host is set up at client start, a
disable/enable change takes effect on the client's next launch.

## Not to be confused with `mcp_servers.toml`

The daemon's `mcp_servers.toml` configures tools that run **with the daemon**
(in the pod, for a cluster deployment). `client-mcp.toml` configures tools that
run **on the edge**. Same schema, opposite locality.
