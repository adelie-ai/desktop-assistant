# Adele - Linux AI Platform

A Rust AI platform for Linux desktops and applications, with:
- D-Bus API for conversation lifecycle and streaming responses
- Multiple LLM backends (`ollama`, `openai`, `anthropic`)
- MCP tool integration over stdio
- Optional terminal UI (TUI) client

## Project Status

Much of this codebase is currently AI-generated, and it has not yet been comprehensively reviewed by humans. It appears to work well in practice, but it should still be treated as experimental.

The current phase of the project is focused on mapping the landscape and getting core functionality in place, so this experimental status is expected to remain for a while.

Community feedback and contributions are very welcome as the platform matures.

The assistant persona is named **Adele**, in reference to the **Adélie penguin**.

## Workspace at a Glance

- `crates/core` — domain model + ports (hexagonal core)
- `crates/dbus-interface` — D-Bus adapter for conversation service
- `crates/daemon` — runtime wiring and D-Bus service host
- `crates/llm-openai` — streaming OpenAI-compatible client
- `crates/mcp-client` — MCP process client + tool executor
- `crates/tui` — interactive terminal client

## Desktop Integrations

- KDE widgets (Plasmoids) are provided in this repository.
- A KDE app/integration surface is also provided for interacting with the assistant.
- GNOME, COSMIC, and generic desktop integration support are planned.

## Integration Model

- This project is intended to be an AI platform with integration points for desktop environments and applications, not only a standalone desktop assistant.
- The platform exposes extensive D-Bus-based APIs for integration with desktop environments and applications.
- The platform makes extensive use of MCP services for pluggable (and un-pluggable) functionality.

## Privacy and Connectivity

The system is designed for privacy first, while still offering cloud LLM connectors as a pragmatic option.

If you use Ollama, the assistant can run entirely offline, preserving privacy. In practice, strong offline quality usually requires larger models and suitable hardware.

If local hardware is limited, cloud services may currently provide better results when that tradeoff is acceptable to you. Nothing in the assistant architecture inherently requires cloud services, and as hardware becomes cheaper over time, fully local operation is expected to become the default for more users.

## Requirements

- Rust (stable, edition 2024)
- Linux session D-Bus (`DBUS_SESSION_BUS_ADDRESS` available)
- For cloud connectors, a connector-specific API key (for example `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- Optional MCP servers (for tools)

## Quick Start

### 1) Build

```bash
cargo build --workspace
```

### 2) Configure connector

Default connector is `openai`.

To opt into local Ollama instead, set `llm.connector = "ollama"` in your daemon config (`$XDG_CONFIG_HOME/desktop-assistant/daemon.toml`, or `~/.config/desktop-assistant/daemon.toml`).

For cloud connectors, export the matching API key environment variable:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# optional connector overrides:
export OPENAI_MODEL=gpt-4o
export OPENAI_BASE_URL=https://api.openai.com/v1
```

Connector key naming convention is generic:
- Secret backend account key defaults to `<connector>_api_key`.
- Environment fallback defaults to `<CONNECTOR>_API_KEY`.
- Connector names are normalized to alphanumeric/underscore (for example, `aws-bedrock` → `aws_bedrock_api_key` and `AWS_BEDROCK_API_KEY`).

Secret backend default is `auto`:

- `SetApiKey` writes to a DE-agnostic local file store: `$XDG_DATA_HOME/desktop-assistant/secrets/<connector>_api_key` (or `~/.local/share/desktop-assistant/secrets/...`).
- Reads check that file store first.
- If missing there, reads try systemd credentials (`$CREDENTIALS_DIRECTORY`).
- If still missing, reads fall back to desktop keyring backends (`libsecret`/`kwallet`).
- Environment variables remain the final fallback.

To provide desktop-agnostic secrets with systemd user services, add a drop-in override:

```bash
mkdir -p ~/.config/systemd/user/desktop-assistant-daemon.service.d
cat > ~/.config/systemd/user/desktop-assistant-daemon.service.d/credentials.conf <<'EOF'
[Service]
# one file per connector account key
LoadCredential=openai_api_key:%h/.config/desktop-assistant/credentials/openai_api_key
LoadCredential=anthropic_api_key:%h/.config/desktop-assistant/credentials/anthropic_api_key
EOF
systemctl --user daemon-reload
systemctl --user restart desktop-assistant-daemon
```

For development service use the same pattern with `desktop-assistant-daemon-dev.service.d`.

### 3) (Optional) Configure MCP servers

Create `~/.config/desktop-assistant/mcp_servers.toml` (or under `$XDG_CONFIG_HOME`):

```toml
[[servers]]
name = "fileio"
command = "fileio-mcp"
args = ["serve", "--mode", "stdio"]
```

### 3b) (Optional) Git persistence for memories/preferences

To version built-in memory and preferences locally, enable git persistence:

```toml
[persistence.git]
enabled = true
```

With this mode, updates are committed to a git repo in your assistant data directory (`$XDG_DATA_HOME/desktop-assistant`, or `~/.local/share/desktop-assistant`).

To also push each update to a remote backup:

```toml
[persistence.git]
enabled = true
remote_url = "git@github.com:you/assistant-memory.git"
remote_name = "origin"
push_on_update = true
```

### 4) Run daemon

```bash
cargo run -p desktop-assistant-daemon
```

### 5) Run TUI client (separate terminal)

```bash
cargo run -p desktop-assistant-tui
```

## Service Setup (systemd user + just)

Install the user service unit + D-Bus activation mapping and reload systemd:

```bash
just install-service
```

With that installed, any client D-Bus method call to `org.desktopAssistant` can auto-start the daemon.

Enable and start backend on login:

```bash
just backend-enable
```

If you only want on-demand startup (no login auto-start), skip `backend-enable`.

Common service operations:

```bash
just backend-status
just backend-restart
just backend-logs
```

Run a development daemon in parallel with the regular user service (separate D-Bus name):

```bash
just dev-backend
```

Or install a dedicated user systemd service for development mode (plus activation mapping):

```bash
just install-service-dev
just backend-dev-enable
```

Common dev service operations:

```bash
just backend-dev-status
just backend-dev-restart
just backend-dev-logs
```

Run TUI against that development daemon:

```bash
just dev-frontend
```

In either chat widget, set **Mode** to **Development** to make panel/desktop widgets target `org.desktopAssistant.Dev`.

### Activation Troubleshooting

If D-Bus calls return "The name is not activatable":

```bash
just install-service
just install-service-dev
```

Check that session bus activation entries exist:

```bash
gdbus call --session \
	--dest org.freedesktop.DBus \
	--object-path /org/freedesktop/DBus \
	--method org.freedesktop.DBus.ListActivatableNames \
	| grep -Eo 'org\.desktopAssistant(\.Dev)?' | sort -u
```

Expected output includes:

- `org.desktopAssistant`
- `org.desktopAssistant.Dev`

Force a clean activation test (service should transition from inactive to active after the call):

```bash
systemctl --user stop desktop-assistant-daemon
echo before=$(systemctl --user is-active desktop-assistant-daemon 2>/dev/null || true)
gdbus call --session --dest org.desktopAssistant --object-path /org/desktopAssistant/Settings --method org.desktopAssistant.Settings.GetLlmSettings
echo after=$(systemctl --user is-active desktop-assistant-daemon 2>/dev/null || true)
```

If activatable names still do not appear, reload both managers and re-check:

```bash
systemctl --user daemon-reload
gdbus call --session --dest org.freedesktop.DBus --object-path /org/freedesktop/DBus --method org.freedesktop.DBus.ReloadConfig
```

## KDE Widgets (Plasmoids)

This repository includes two KDE Plasma widgets that talk to the daemon over D-Bus:

- Panel widget: `kde/plasmoid/org.desktopassistant.panelchat`
- Desktop widget: `kde/plasmoid/org.desktopassistant.desktopchat`

Install both for your user:

```bash
kpackagetool6 --type Plasma/Applet --install kde/plasmoid/org.desktopassistant.panelchat
kpackagetool6 --type Plasma/Applet --install kde/plasmoid/org.desktopassistant.desktopchat
```

Upgrade after local changes:

```bash
kpackagetool6 --type Plasma/Applet --upgrade kde/plasmoid/org.desktopassistant.panelchat
kpackagetool6 --type Plasma/Applet --upgrade kde/plasmoid/org.desktopassistant.desktopchat
```

Usage:

- Add **Desktop Assistant** to the panel/task bar for quick popup chat.
- Add **Desktop Assistant (Desktop)** to the desktop for an always-visible chat card.
- Click **Settings** in chat widgets to open **System Settings → Desktop Assistant** for connector/model/search configuration.
- Widget controls include:
	- **New**: start a fresh conversation.
	- **Debug**: show/hide low-level tool execution status lines.
	- **Clear**: clear the visible transcript without deleting conversation history.

Notes:

- Both chat widgets include a **service selector** (Production/Development) and call the selected D-Bus service at `/org/desktopAssistant/Conversations`.
- Widgets auto-detect whether `org.desktopAssistant.Dev` currently has an owner on the session bus.
- If the dev environment is not running, chat widgets hide themselves.
- Both widgets shell out to `python3` and `gdbus` to call methods documented in `docs/dbus-api.md`.
- API keys are write-only over D-Bus (`SetApiKey` only) and are never returned to clients.
- Daemon can auto-start on first D-Bus method call once `just install-service` is set up.

## KDE System Settings Panel (KCM)

Build the Desktop Assistant KCM module:

```bash
just kcm-build
```

Install it user-locally (development copy):

```bash
just kcm-install
```

Install system-wide (recommended for daily use + KDE discovery, requires sudo):

```bash
just kcm-install-system
```

Refresh cache and verify discovery:

```bash
just kcm-refresh
```

Open using the user-local plugin environment:

```bash
just kcm-open
```

Open using system plugin paths:

```bash
just kcm-open-system
```

### KCM install modes (important)

KDE can see both a **system** KCM install (`/usr/...`) and a **user-local** KCM install (`~/.local/...`).
If both exist, it can look like settings/UI changes are "randomly" reverting depending on which one is loaded.

Choose one mode and stick to it:

- **System mode (recommended for stability):**
	- Install/update with `just kcm-install-system`
	- Open with `just kcm-open-system`
	- Do not keep a user-local KCM copy installed
- **Local mode (recommended for active KCM development):**
	- Install/update with `just kcm-install`
	- Open with `just kcm-open`
	- Do not keep a system KCM copy installed

### One-shot cleanup commands

- Remove only **local** KCM artifacts (keeps system install intact):

```bash
just kcm-cleanup
```

- Remove only **system** KCM artifacts (requires sudo):

```bash
just kcm-cleanup-system
```

- Then refresh cache:

```bash
just kcm-refresh
```

### Recovery playbook (if UI looks wrong or old)

1. Pick target mode (**system** or **local**).
2. Remove the other mode's artifacts using the cleanup command above.
3. Reinstall target mode (`just kcm-install-system` or `just kcm-install`).
4. Refresh (`just kcm-refresh`).
5. Open with matching command (`just kcm-open-system` or `just kcm-open`).

Note: KDE loads KCM plugins from Qt6 plugin paths (for example `/usr/lib64/qt6/plugins` and `~/.local/lib64/qt6/plugins`).

After install, open KDE System Settings and search for **Desktop Assistant**.
You can also launch directly:

```bash
kcmshell6 kcm_desktopassistant
```

## Core Commands

```bash
# format
cargo fmt

# tests
cargo test --workspace

# strict linting
cargo clippy --workspace --all-targets -- -D warnings
```

## Packaging

```bash
# Docker-friendly package builds (deb, rpm, flatpak)
just package-all-docker

# Snap package build (run on host with snapd/core24 available)
just package-snap
```

Note: `package-all-docker` intentionally excludes Snap because Snap builds for `base: core24`
require a working `snapd`/`core24` runtime that is not reliable inside Docker/Podman container builds.

## Documentation

- [Architecture](docs/architecture.md)
- [D-Bus API](docs/dbus-api.md)
- [MCP Integration](docs/mcp-integration.md)
- [Development Guide](docs/development.md)

## Built-in Memory Tools

The daemon now includes built-in in-process tools exposed through the MCP executor, even when no external MCP servers are configured:

- Preferences:
	- `builtin_preferences_remember`
	- `builtin_preferences_search`
	- `builtin_preferences_retrieve`
- Factual memory:
	- `builtin_memory_remember`
	- `builtin_memory_search`
	- `builtin_memory_retrieve`
	- `builtin_memory_update`

Storage paths:

- Preferences:
	- `$XDG_DATA_HOME/desktop-assistant/preferences.json`, or
	- `~/.local/share/desktop-assistant/preferences.json` when `XDG_DATA_HOME` is unset.
- Factual memory:
	- `$XDG_DATA_HOME/desktop-assistant/factual_memory.json`, or
	- `~/.local/share/desktop-assistant/factual_memory.json` when `XDG_DATA_HOME` is unset.

## Notes

- If a cloud connector is selected and its API key is missing, daemon still starts but prompt calls fail at runtime.
- If MCP config is missing, daemon runs with no external tools.
- Daemon LLM settings are read from:
	- `$XDG_CONFIG_HOME/desktop-assistant/daemon.toml`, or
	- `~/.config/desktop-assistant/daemon.toml` if `XDG_CONFIG_HOME` is unset.
- API keys can be stored in the desktop keyring via `libsecret`/Secret Service (default backend).
- KDE Wallet remains supported via `llm.secret.backend = "kwallet"` in `daemon.toml`.
- Conversations persist across daemon restarts in:
	- `$XDG_DATA_HOME/desktop-assistant/conversations.json`, or
	- `~/.local/share/desktop-assistant/conversations.json` if `XDG_DATA_HOME` is unset.

## License

Desktop Assistant is licensed under **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`).
See the [LICENSE](LICENSE) file for the full text.
