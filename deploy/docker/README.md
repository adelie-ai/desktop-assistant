# desktop-assistant container image

One multi-stage image (`Containerfile` at the repo root) carries the
orchestrator daemon, the D-Bus bridge, and the full MCP server fleet behind a
single role-selecting entrypoint.

## Build

```sh
# from the repo root
podman build -f Containerfile -t desktop-assistant .
# (docker build … also works)
```

The build runs three stages:

1. compiles the workspace binaries `desktop-assistant-daemon` and
   `adelie-dbus-bridge`;
2. `cargo install`s the MCP fleet from their individual repos into `/opt/mcp`
   (needs network access at build time);
3. assembles a slim `debian:bookworm-slim` runtime with all binaries on
   `PATH`, running as the unprivileged `assistant` user.

## Roles

The first argument selects what the container runs (see
`deploy/docker/entrypoint.sh`):

| Command                                            | Runs                          |
| -------------------------------------------------- | ----------------------------- |
| `podman run … desktop-assistant`                   | `desktop-assistant-daemon` (default) |
| `podman run … desktop-assistant daemon`            | `desktop-assistant-daemon`    |
| `podman run … desktop-assistant bridge`            | `adelie-dbus-bridge`          |
| `podman run … desktop-assistant fileio-mcp --help` | runs the arg verbatim         |

Any unrecognised first argument is exec'd directly, so individual MCP servers
and a debugging shell are reachable without a custom command override.

## Baked MCP fleet

Twelve servers are installed and on `PATH`, each invoked by the daemon as
`<bin> serve --mode stdio`:

```
fileio-mcp        terminal-mcp     tasks-mcp        timeclock-mcp
web-mcp           cve-mcp          geocode-mcp      openstreetmap-mcp
weather-forecast-mcp               internet-radio-mcp
skills-mcp        gen-mcp
```

The fleet list lives in the `MCP_SERVERS` build ARG in `Containerfile` (single
source of truth). The installed binary names must match the `command` values in
`mcp_servers.toml`. The daemon degrades gracefully if a given tool server is
absent.

**Reproducibility:** the servers currently track each repo's default branch. A
`MCP_REF` build ARG is wired up to pin every repo to a single commit/tag once
the fleet is versioned in lockstep — see the TODO in `Containerfile`.

## Networking / config

- `EXPOSE 11339` and `DESKTOP_ASSISTANT_WS_BIND=0.0.0.0:11339` are set, but the
  WebSocket front door is **not** enabled in the image. The deployment
  (compose/k8s) opts in via `DESKTOP_ASSISTANT_WS_ENABLED`.
- D-Bus is optional: `DESKTOP_ASSISTANT_DBUS_REQUIRED=false`.
- XDG dirs live under `/home/assistant`; mount a volume there to persist state.
