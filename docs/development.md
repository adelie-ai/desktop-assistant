# Development Guide

The assistant persona is named **Adele**, in reference to the **Adélie penguin**.

## Day-to-day Commands

```bash
# format
cargo fmt

# full test suite
cargo test --workspace

# strict lints
cargo clippy --workspace --all-targets -- -D warnings
```

## Run Components

```bash
# daemon
cargo run -p desktop-assistant-daemon

# tui
cargo run -p desktop-assistant-tui
```

## Run Daemon in Docker

```bash
docker build -t desktop-assistant-daemon .

# exposes ws://localhost:11339/ws from container
docker run --rm -p 11339:11339 \
  -e OPENAI_API_KEY=your_key_here \
  desktop-assistant-daemon
```

Container defaults from `Dockerfile`:
- `DESKTOP_ASSISTANT_WS_BIND=0.0.0.0:11339`
- `DESKTOP_ASSISTANT_DBUS_REQUIRED=false` (daemon will continue if session D-Bus is unavailable)

For systemd user service + D-Bus activation setup:

```bash
just install-service
```

After install, method calls to `org.desktopAssistant` can auto-start the daemon if it is not already running.

## Activation Troubleshooting

Use the canonical troubleshooting checklist in [README.md](README.md#activation-troubleshooting).

## Environment

Quick provider privacy + console links: [cloud-providers.md](cloud-providers.md)

Default connector is `openai`.

To opt into local Ollama, set `llm.connector = "ollama"` in `$XDG_CONFIG_HOME/desktop-assistant/daemon.toml` (or `~/.config/desktop-assistant/daemon.toml`).

For cloud connectors, set connector credentials:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
export AWS_REGION=us-east-1
# optional single-field Bedrock credentials format:
# export AWS_BEDROCK_API_KEY=ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]
```

For Bedrock, set `llm.connector = "bedrock"` (or `"aws-bedrock"`) and use `llm.base_url` as region (for example `us-east-1`) or a Bedrock runtime endpoint URL.

Bedrock credentials use the standard AWS SDK credential provider chain, so local development should normally use configured AWS CLI credentials/profile (`aws configure` or `aws configure sso`) with Bedrock permissions in the target region.

Connector key naming convention is generic:
- Secret backend account key defaults to `<connector>_api_key`.
- Environment fallback defaults to `<CONNECTOR>_API_KEY`.
- Connector names are normalized to alphanumeric/underscore (for example, `aws-bedrock` → `aws_bedrock_api_key` and `AWS_BEDROCK_API_KEY`).

Secret backend default is `auto`:
- `SetApiKey` writes to `$XDG_DATA_HOME/desktop-assistant/secrets/<connector>_api_key` (or `~/.local/share/desktop-assistant/secrets/...`).
- Reads check that file first, then systemd credentials, then desktop keyrings, then environment variables.

WebSocket auth uses bearer JWTs:
- Generate a token over D-Bus with `GenerateWsJwt(subject)`.
- Connect to `/ws` with `Authorization: Bearer <token>`.
- Tokens are locally signed by the daemon and multiple tokens can coexist until expiry.

TUI transport defaults to WebSocket and can be configured:

```bash
# command-line flags (via clap)
desktop-assistant-tui --transport ws --ws-url ws://127.0.0.1:11339/ws
desktop-assistant-tui --transport dbus
desktop-assistant-tui --ws-jwt eyJ... --ws-subject desktop-tui

# default (if unset): ws
export DESKTOP_ASSISTANT_TUI_TRANSPORT=ws

# override websocket endpoint (default: ws://127.0.0.1:11339/ws)
export DESKTOP_ASSISTANT_TUI_WS_URL=ws://127.0.0.1:11339/ws

# optional: provide JWT directly (skips D-Bus bootstrap)
export DESKTOP_ASSISTANT_TUI_WS_JWT=eyJ...

# optional: subject used when TUI asks D-Bus to mint a JWT
export DESKTOP_ASSISTANT_TUI_WS_SUBJECT=desktop-tui

# force legacy local D-Bus transport
export DESKTOP_ASSISTANT_TUI_TRANSPORT=dbus
```

KDE widget helper now supports named connections (defaulting to `local` D-Bus unless changed in KCM):

```bash
# optional: pin a widget or helper call to a named connection
export DESKTOP_ASSISTANT_WIDGET_CONNECTION=my-cluster

# optional explicit token for remote clusters (skips local D-Bus JWT bootstrap)
export DESKTOP_ASSISTANT_WIDGET_WS_JWT=eyJ...
```

Legacy overrides are still available:

```bash
export DESKTOP_ASSISTANT_WIDGET_TRANSPORT=ws
export DESKTOP_ASSISTANT_WIDGET_WS_URL=ws://127.0.0.1:11339/ws
export DESKTOP_ASSISTANT_WIDGET_WS_SUBJECT=desktop-widget
export DESKTOP_ASSISTANT_WIDGET_DBUS_SERVICE=org.desktopAssistant
```

For a desktop-agnostic setup, prefer systemd credentials via user-service drop-ins:

```bash
mkdir -p ~/.config/systemd/user/desktop-assistant-daemon.service.d
cat > ~/.config/systemd/user/desktop-assistant-daemon.service.d/credentials.conf <<'EOF'
[Service]
LoadCredential=openai_api_key:%h/.config/desktop-assistant/credentials/openai_api_key
LoadCredential=anthropic_api_key:%h/.config/desktop-assistant/credentials/anthropic_api_key
EOF
systemctl --user daemon-reload
systemctl --user restart desktop-assistant-daemon
```

Optional:

```bash
export OPENAI_MODEL=gpt-5.2
export OPENAI_BASE_URL=https://api.openai.com/v1
export RUST_LOG=info
```

If `OPENAI_MODEL` is not set, the daemon defaults OpenAI to `gpt-5.2`.

To enable local git versioning for built-in memory/preferences:

```toml
[persistence.git]
enabled = true
```

To push updates to a remote:

```toml
[persistence.git]
enabled = true
remote_url = "git@github.com:you/assistant-memory.git"
remote_name = "origin"
push_on_update = true
```

## Testing MCP Integration

- E2E tests may require external binaries (`fileio-mcp`, `python3`)
- Tests skip gracefully if optional tools are unavailable

Useful targeted runs:

```bash
cargo test -p desktop-assistant-mcp-client
cargo test -p desktop-assistant-mcp-client --test e2e_fileio
cargo test -p desktop-assistant-mcp-client --test e2e_dynamic_list_changed
```

## Project Conventions

- Keep core logic in `crates/core` independent of adapters
- Prefer trait-based boundaries over direct dependency coupling
- Keep docs and tests updated when interfaces change
