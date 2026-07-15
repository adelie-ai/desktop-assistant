# Composable daemon + MCP-fleet base image

`Dockerfile.fleet` (repo root) builds the **composable base image**: the
`desktop-assistant` daemon **plus** the full MCP-server fleet, with a curated
default config seeded on first boot. This is the image to deploy, and the one to
derive new images from.

> Every registry, host, and namespace below is a **placeholder** — substitute
> your own and keep real values out of git.

## What's in it

Stable on-disk layout (the contract downstream images build on):

| Path | What |
| --- | --- |
| `/usr/local/bin/desktop-assistant-daemon` | the daemon (on `$PATH`) |
| `/opt/adele/mcp/<name>-mcp` | bundled fleet binaries (referenced by **absolute path** — the daemon spawns via `Command::new` with no `$PATH` augmentation) |
| `/opt/adele/mcp_servers.default.toml` | curated default config, owned by the daemon user (uid 10001) |

On first boot the daemon seeds `/opt/adele/mcp_servers.default.toml` into its
config (`DESKTOP_ASSISTANT_MCP_DEFAULT_CONFIG`, issue #491) **only when no
`mcp_servers.toml` exists yet** — runtime edits from the settings UI win and
persist. On Kubernetes the config lives on the `/state` PVC, so the seed happens
once and survives restarts.

## The bundled fleet

Curated in `deploy/mcp/mcp_servers.default.toml` by **danger x usefulness**:

- **Enabled** (safe, useful, zero config): `weather-forecast`, `geocode`,
  `openstreetmap`, `cve`, `tasks`, `timeclock`, `skills`.
- **Configured-but-disabled** (opt-in): `terminal` (shell exec), `command`
  (needs a `--config`), `fileio` (filesystem writes), `homeassistant` (needs a
  URL + token), `web` (needs Chromium), `internet-radio` (needs `mpv` + audio).

Enable a disabled server from the settings UI (or by editing the config), after
providing whatever it needs.

## Build

The daemon and the fleet servers path-build from sibling source trees, so the
build context is a **staged dir** with `desktop-assistant/` and the 13 `*-mcp`
repos as siblings. Stage clean copies (no `target/`, `.git/`, packaging cruft):

```sh
# ADELE = your adelie-ai checkout root (holds the sibling repos).
ADELE=<path-to-your-adelie-ai-checkout>
CTX=$(mktemp -d)/fleet-ctx
mkdir -p "$CTX"
for r in desktop-assistant command-mcp cve-mcp fileio-mcp geocode-mcp \
         homeassistant-mcp internet-radio-mcp openstreetmap-mcp skills-mcp \
         tasks-mcp terminal-mcp timeclock-mcp weather-forecast-mcp web-mcp; do
  rsync -aL --exclude target --exclude .git --exclude build \
        --exclude '.flatpak-builder' --exclude .venv --exclude .worktrees \
        "$ADELE/$r/" "$CTX/$r/"
done

REG=registry.example.com:5000                      # your registry (do NOT commit)
IMG="$REG/adele/adele-daemon:fleet-$(git -C "$ADELE/desktop-assistant" rev-parse --short HEAD)"
podman build -t "$IMG" -f "$CTX/desktop-assistant/Dockerfile.fleet" "$CTX"
podman push "$IMG"
```

## Derive a new image

Add or swap servers with no base rebuild:

```dockerfile
FROM registry.example.com:5000/adele/adele-daemon:fleet-<tag>

# Add a new server binary (build it however you like, then drop it in):
COPY my-new-mcp /opt/adele/mcp/my-new-mcp

# Replace the shipped default so the new server is present on first boot:
COPY mcp_servers.toml /opt/adele/mcp_servers.default.toml
```

Enable a dependency-needing server, e.g. `web` (needs Chromium):

```dockerfile
FROM registry.example.com:5000/adele/adele-daemon:fleet-<tag>
USER root
RUN apt-get update && apt-get install -y --no-install-recommends chromium \
    && rm -rf /var/lib/apt/lists/*
USER assistant
# then set `web` enabled = true (args = ["serve", "--chrome-arg=--no-sandbox"])
# in the config you COPY over /opt/adele/mcp_servers.default.toml.
```

The default config in your derived image must keep the same rules: `command`
values are **absolute** `/opt/adele/mcp/<name>` paths, and the file is owned by
uid 10001 (`COPY --chown=10001:10001`) so the daemon can enforce 0600 on read.
