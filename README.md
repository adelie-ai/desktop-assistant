# Desktop Assistant

A Rust desktop assistant with:
- D-Bus API for conversation lifecycle and streaming responses
- OpenAI-compatible LLM backend
- MCP tool integration over stdio
- Optional terminal UI (TUI) client

## Workspace at a Glance

- `crates/core` — domain model + ports (hexagonal core)
- `crates/dbus-interface` — D-Bus adapter for conversation service
- `crates/daemon` — runtime wiring and D-Bus service host
- `crates/llm-openai` — streaming OpenAI-compatible client
- `crates/mcp-client` — MCP process client + tool executor
- `crates/tui` — interactive terminal client

## Requirements

- Rust (stable, edition 2024)
- Linux session D-Bus (`DBUS_SESSION_BUS_ADDRESS` available)
- `OPENAI_API_KEY` for real LLM calls
- Optional MCP servers (for tools)

## Quick Start

### 1) Build

```bash
cargo build --workspace
```

### 2) Configure OpenAI

```bash
export OPENAI_API_KEY=your_key_here
# optional:
export OPENAI_MODEL=gpt-4o
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3) (Optional) Configure MCP servers

Create `~/.config/desktop-assistant/mcp_servers.toml` (or under `$XDG_CONFIG_HOME`):

```toml
[[servers]]
name = "fileio"
command = "fileio-mcp"
args = ["serve", "--mode", "stdio"]
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

Install the user service unit and reload systemd:

```bash
just install-service
```

Enable and start backend on login:

```bash
just backend-enable
```

Common service operations:

```bash
just backend-status
just backend-restart
just backend-logs
```

## KDE Widgets (Plasmoids)

This repository includes two KDE Plasma widgets that talk to the daemon over D-Bus:

- Panel widget: `kde/plasmoid/org.desktopassistant.panelchat`
- Desktop widget: `kde/plasmoid/org.desktopassistant.desktopchat`
- Settings widget: `kde/plasmoid/org.desktopassistant.settings`

Install both for your user:

```bash
kpackagetool6 --type Plasma/Applet --install kde/plasmoid/org.desktopassistant.panelchat
kpackagetool6 --type Plasma/Applet --install kde/plasmoid/org.desktopassistant.desktopchat
kpackagetool6 --type Plasma/Applet --install kde/plasmoid/org.desktopassistant.settings
```

Upgrade after local changes:

```bash
kpackagetool6 --type Plasma/Applet --upgrade kde/plasmoid/org.desktopassistant.panelchat
kpackagetool6 --type Plasma/Applet --upgrade kde/plasmoid/org.desktopassistant.desktopchat
kpackagetool6 --type Plasma/Applet --upgrade kde/plasmoid/org.desktopassistant.settings
```

Usage:

- Add **Desktop Assistant** to the panel/task bar for quick popup chat.
- Add **Desktop Assistant (Desktop)** to the desktop for an always-visible chat card.
- Add **Desktop Assistant Settings** to configure connector/model/base URL and API key.
- Widget controls include:
	- **New**: start a fresh conversation.
	- **Debug**: show/hide low-level tool execution status lines.
	- **Clear**: clear the visible transcript without deleting conversation history.

Notes:

- Both widgets use the service `org.desktopAssistant` at `/org/desktopAssistant/Conversations`.
- Both widgets shell out to `python3` and `gdbus` to call methods documented in `docs/dbus-api.md`.
- Settings widget uses `org.desktopAssistant.Settings` D-Bus methods.
- API keys are write-only over D-Bus (`SetApiKey` only) and are never returned to clients.
- Ensure the daemon is running (`just backend-status` / `just backend-restart`) before sending prompts.

## KDE System Settings Panel (KCM)

Build the Desktop Assistant KCM module:

```bash
just kcm-build
```

Install it user-locally:

```bash
just kcm-install
```

Install system-wide (recommended for normal KDE discovery, requires sudo):

```bash
just kcm-install-system
```

Refresh cache and verify discovery:

```bash
just kcm-refresh
```

Open directly from shell with the required user-local plugin environment:

```bash
just kcm-open
```

Note: KDE loads KCM plugins from Qt6 plugin paths (for example `/usr/lib64/qt6/plugins`).
The `just` recipes install to that location to ensure `kcmshell6` can find the module.

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

- If `OPENAI_API_KEY` is missing, daemon still starts but prompt calls will fail at runtime.
- If MCP config is missing, daemon runs with no external tools.
- Daemon LLM settings are read from:
	- `$XDG_CONFIG_HOME/desktop-assistant/daemon.toml`, or
	- `~/.config/desktop-assistant/daemon.toml` if `XDG_CONFIG_HOME` is unset.
- API keys can be stored in the desktop keyring via `libsecret`/Secret Service (default backend).
- KDE Wallet remains supported via `llm.secret.backend = "kwallet"` in `daemon.toml`.
- Conversations persist across daemon restarts in:
	- `$XDG_DATA_HOME/desktop-assistant/conversations.json`, or
	- `~/.local/share/desktop-assistant/conversations.json` if `XDG_DATA_HOME` is unset.
