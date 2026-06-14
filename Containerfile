# syntax=docker/dockerfile:1
#
# Multi-stage build for the desktop-assistant container image.
#
# The resulting image carries THREE things behind one role-selecting entrypoint:
#   1. desktop-assistant-daemon  — the orchestrator daemon (default role)
#   2. adelie-dbus-bridge        — the D-Bus bridge (role "bridge")
#   3. the full MCP server fleet — invoked by the daemon as child processes
#
# Pick the role at runtime, e.g.:
#   podman run … <image>            # → daemon (default CMD)
#   podman run … <image> bridge     # → adelie-dbus-bridge
#   podman run … <image> fileio-mcp serve --mode stdio   # → run a tool verbatim
#
# Build with `podman build -f Containerfile .` (or `docker build`).

# ---------------------------------------------------------------------------
# Stage 1: build the workspace binaries (daemon + bridge).
# ---------------------------------------------------------------------------
# rust:1-bookworm tracks the latest 1.x; the workspace pins the exact compiler
# via rust-toolchain.toml (currently 1.96.0), which rustup honours on first
# cargo invocation, so the channel here only needs to be recent enough.
FROM rust:1-bookworm AS builder

WORKDIR /workspace

# The daemon FFI-links libpam (crates/daemon/src/config/pam_auth.rs) for the
# WS local-system password auth path — it is NOT feature-gated, so `-lpam` is
# required at link time even when that auth method is disabled at runtime.
# libpam0-dev provides the linkable libpam.so.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpam0g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Build only the two binaries we ship; --locked keeps Cargo.lock authoritative.
RUN cargo build --release --locked \
    --bin desktop-assistant-daemon \
    --bin adelie-dbus-bridge

# ---------------------------------------------------------------------------
# Stage 2: install the MCP server fleet from their individual repos.
# ---------------------------------------------------------------------------
# Each server lives in its own repo under https://github.com/adelie-ai/<name>
# and is installed into a self-contained prefix (/opt/mcp) so the runtime stage
# can grab them all with a single COPY of /opt/mcp/bin/*.
#
# NOTE: these `cargo install --git` invocations require network/build access at
# image-build time. The daemon degrades gracefully if a given tool server is
# absent, so a single failing repo is not fatal to running the daemon — but a
# clean build expects all twelve present.
FROM rust:1-bookworm AS mcp-builder

# The fleet list is kept in ONE place. The installed binary names MUST match the
# `command` values in mcp_servers.toml.
# TODO(reproducibility): pin each repo to a tag/commit. `cargo install --git`
# accepts `--tag`/`--rev`/`--branch`; the simplest path is a single MCP_REF ARG
# applied to every repo once the fleet is tagged in lockstep, or per-repo ARGs
# if they version independently. Until then this tracks each repo's default branch.
ARG MCP_REF=
ARG MCP_SERVERS="fileio-mcp terminal-mcp tasks-mcp timeclock-mcp web-mcp cve-mcp geocode-mcp openstreetmap-mcp weather-forecast-mcp internet-radio-mcp skills-mcp gen-mcp"

RUN set -eux; \
    ref_flag=""; \
    if [ -n "${MCP_REF}" ]; then ref_flag="--rev ${MCP_REF}"; fi; \
    for srv in ${MCP_SERVERS}; do \
        cargo install \
            --git "https://github.com/adelie-ai/${srv}" \
            ${ref_flag} \
            --locked \
            --root /opt/mcp; \
    done

# ---------------------------------------------------------------------------
# Stage 3: slim runtime image.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

# ca-certificates for outbound TLS (Bedrock, MCP HTTP fetches, etc.).
# libpam0 is the runtime shared lib the daemon dynamically links (see builder
# note above); without it the daemon fails to start with a loader error.
# /bin/sh is already present in the base image and is intentionally kept —
# terminal-mcp shells out via `sh -c`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libpam0g \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user with a real home for the XDG dirs below.
RUN useradd --create-home --uid 10001 assistant
WORKDIR /home/assistant

# Workspace binaries.
COPY --from=builder /workspace/target/release/desktop-assistant-daemon /usr/local/bin/desktop-assistant-daemon
COPY --from=builder /workspace/target/release/adelie-dbus-bridge /usr/local/bin/adelie-dbus-bridge

# MCP fleet binaries (all under /opt/mcp/bin from the cargo install --root).
COPY --from=mcp-builder /opt/mcp/bin/ /usr/local/bin/

# Role-selecting entrypoint.
COPY deploy/docker/entrypoint.sh /usr/local/bin/adelie-entrypoint
RUN chmod +x /usr/local/bin/adelie-entrypoint

# Environment defaults (compose/k8s may override). WS_ENABLED is deliberately
# NOT set here — the orchestrator only listens on the WS bind when the
# deployment opts in via DESKTOP_ASSISTANT_WS_ENABLED.
ENV RUST_LOG=info
ENV DESKTOP_ASSISTANT_WS_BIND=0.0.0.0:11339
ENV DESKTOP_ASSISTANT_DBUS_REQUIRED=false
ENV XDG_CONFIG_HOME=/home/assistant/.config
ENV XDG_DATA_HOME=/home/assistant/.local/share
ENV XDG_STATE_HOME=/home/assistant/.local/state
ENV XDG_CACHE_HOME=/home/assistant/.cache

EXPOSE 11339
USER assistant

ENTRYPOINT ["adelie-entrypoint"]
CMD ["daemon"]
