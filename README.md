# Adelie Linux AI Platform

An API-first AI platform for Linux desktops and applications, with full-featured D-Bus and Websocket API

Provides the **Adele** desktop assistant.

## Features

- Automated knowledge base maintenance and the ability to introspect memory
- Dream cycle to consolidate knowledge base and learn from conversations
- Support for long tool lists with a vectorized tool search.
- Storage in Postgres, with vectorized knowledge base and tool search using `pgvector`
- Use different connectors and models for primary interactions, backend tasks, and search vectorization

## Integrations
- Multiple LLM backends (`ollama`, `openai`, `anthropic`, `aws bedrock`)
- MCP tool integration over stdio with several MCP servers provided by the `adelie-mcp` project
- `ratatui`-based TUI client
- `gtk-client` standalone GTK-based client
- KDE plasmoids and control panel, with other DEs to come

## Project Status

**Much of this codebase is currently AI-generated**, and it has not yet been comprehensively reviewed by humans. It appears to work well in practice, but it should still be treated as **EXPERIMENTAL**.

The current phase of the project is focused on mapping the landscape and getting core functionality in place as quickly as possible, so this experimental status is expected to remain for a while. It's already extremely capable and useful, especially coupled with the MCP servers from the project, but please be careful with it.

Community feedback and contributions are very welcome as the platform matures.

## The Name

The core platform is called the **Adelie** AI platform. The assistant persona implemented on it is named **Adele**.

This branding is fairly superficial and isn't extensively reflected in code at this point, but once it settles, that will probably change. 

## Current AI Connectors

The project currently can use AWS Bedrock, Ollama, OpenAI, and Anthropic APIs.

## Configuration Recommendations

Configuration is a personal thing. You'll want to experiment. Note that there are differences between the connectors in terms of how they do token caching (if they do it at all), and there are connector-specific instructions which help to fine-tune how the model behaves. So they're not all created equally. Try them out if you don't have a hard requirement, and see what works best for you.

### My Setup

I personally configure Adelie with **AWS Bedrock** using the **Anthropic Sonnet and Haiku models** for primary and backend work respectively, and have it use **local ollama** for knowledge base and tool search vectorization. I just find the Anthropic models are better for the type of tasks I give it. Local Ollama for vectorization is great because the models and search requests are small, and much of it happens in the background where the user doesn't know about it.

AWS Bedrock gives me "better" privacy, is faster than Anthropic and OpenAI's APIs, and I don't need to subscribe to another service. I don't have a decent GPU, so local work is mostly out of the question for me, and I can pay for Bedrock for years before I break even buying GPUs. AWS Bedrock DOES carry the risk of costing yourself some cash if something starts looping, but AWS also puts consumption limits that you'll hit by default unless you want to raise them (via AWS support request), so it's not entirely unmanaged risk.

### OpenAI

I initially used the OpenAI connector with GPT 5.3, which worked great. I liked the ability to pay as I go and constrain cost if something ran amuck. I'd just add 10 bucks at a time and know I couldn't accidentally cost myself 1000 bucks if Adelie went haywire. It saved me a couple times early on while I was building the event loops! This is still a good approach to consider. I found that 5.3 worked better than 5.4 for my tasks, but your mileage may vary. 

OpenAI's automatic token caching seems to work really well, but I was not able to get their built-in dynamic tool search working properly. I'm sure it's something silly, but I've punted it for now. 

### Anthropic

The Anthropic connector works, but for some reason seems to burn a lot of tokens and hit daily limits very quickly. I do use the Anthropic models, but via AWS Bedrock. Anthropic's caching doesn't work the same way as OpenAI's, and I think with the dynamic tool lists, it keeps getting invalidated. I wasn't able to get their built-in tool search working (nor for Open AI, for that matter). I've punted these issues for now.

## Integration Model

- This project is intended to be an AI platform with integration points for desktop environments and applications, not only a standalone desktop assistant.
- The platform exposes extensive D-Bus- and Web-socket-based APIs for integration with desktop environments and applications.
- The platform makes extensive use of MCP services for pluggable (and un-pluggable) functionality, and can even write its own MCP servers.

> **MCP servers are essential for real-world usefulness.** Without configured MCP servers, the assistant can hold conversations but cannot take actions (file I/O, task management, time tracking, shell execution, etc.). The built-in memory tools are always present, but meaningful capability comes from external MCP servers.
>
> Companion servers developed for this platform include `fileio-mcp`, `terminal-mcp`, `tasks-mcp`, `timeclock-mcp`, `skills-mcp`, and `calendar-mcp`. See [docs/mcp-services.md](docs/mcp-services.md) for the full list and configuration.

## Privacy and Connectivity

The system is designed for privacy first, while still offering cloud LLM connectors as a pragmatic option. As always, privacy is a choose-your-own-adventure.

If you use Ollama, the assistant can run entirely offline, preserving privacy. In practice, strong offline quality usually requires larger models and suitable hardware.

If local hardware is limited, cloud services may currently provide better results when that tradeoff is acceptable to you. Nothing in the assistant architecture inherently requires cloud services, and as hardware becomes cheaper over time, fully local operation is expected to become the default for more users. 

**Adelie Platform uses API calls to the cloud AI providers. By and large, cloud AI providers do NOT use API call data for training purposes, but you are responsible for understanding the privacy implications of your chosen provider.**

Quick provider privacy + setup links: [docs/cloud-providers.md](docs/cloud-providers.md)

## Requirements

- Rust (stable, edition 2024)
- Linux session D-Bus (`DBUS_SESSION_BUS_ADDRESS` available) for desktop integrations, but service can run websocket-only.
- PostgreSQL with the `pgvector` extension (see below)
- For cloud connectors, connector credentials (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or AWS credentials for Bedrock)
- Optional MCP servers (for tools)

### PostgreSQL setup

The daemon requires a PostgreSQL database for knowledge base storage, tool registry, and conversation history. The `pgvector` extension is required for embedding-based search.

1. Install PostgreSQL and pgvector:

```bash
# Debian/Ubuntu
sudo apt install postgresql postgresql-contrib postgresql-16-pgvector

# Fedora
sudo dnf install postgresql-server postgresql-contrib pgvector_16

# Arch
sudo pacman -S postgresql postgresql-libs
yay -S pgvector  # or install from AUR
```

2. Create a database and user:

```sql
CREATE USER desktop_assistant WITH PASSWORD 'your_password_here';
CREATE DATABASE desktop_assistant OWNER desktop_assistant;
```

3. Enable the vector extension in the database:

```sql
\c desktop_assistant
CREATE EXTENSION IF NOT EXISTS vector;
```

4. Configure the connection in `daemon.toml`:

```toml
[database]
url = "postgres://desktop_assistant:your_password_here@localhost/desktop_assistant"
```

The daemon runs migrations automatically on startup.

## Quick Start

### 1) Build

```bash
cargo build --workspace
```

### 2) Configure connector

Default connector is `openai`.
Default OpenAI model is `gpt-5.4`.

To opt into local Ollama instead, set `llm.connector = "ollama"` in your daemon config (`$XDG_CONFIG_HOME/desktop-assistant/daemon.toml`, or `~/.config/desktop-assistant/daemon.toml`).

For cloud connectors, set credentials for the connector you use:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
export AWS_REGION=us-east-1

# optional connector overrides:
export OPENAI_MODEL=gpt-5.4
export OPENAI_BASE_URL=https://api.openai.com/v1

# optional Bedrock API key field format accepted by this daemon:
# export AWS_BEDROCK_API_KEY=ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]
```

Bedrock notes:
- Set `llm.connector = "bedrock"` (or `"aws-bedrock"`).
- `llm.base_url` is interpreted as AWS region by default (for example `us-east-1`).
- You can also use a Bedrock runtime endpoint URL (for example `https://bedrock-runtime.us-east-1.amazonaws.com`).
- Credentials resolve via the standard AWS SDK credential provider chain (for example `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, shared AWS CLI config from `aws configure`, SSO profiles, or IAM role credentials).
- If you are running locally, configure AWS CLI/profile first (`aws configure` or `aws configure sso`) and ensure the selected profile has Bedrock permissions in the target region.
- Optional connector key format `ACCESS_KEY_ID:SECRET_ACCESS_KEY[:SESSION_TOKEN]` is supported for parity with single-field API key flows.

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

### 3) Configure MCP servers

> **Recommended.** MCP servers provide the tools (file I/O, task management, shell execution, etc.) that make the assistant genuinely useful. See [docs/mcp-services.md](docs/mcp-services.md) for the full list of available servers and their configuration options.

Create `~/.config/desktop-assistant/mcp_servers.toml` (or under `$XDG_CONFIG_HOME`):

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]

[[servers]]
name    = "terminal"
command = "terminal-mcp"
args    = ["serve", "--mode", "stdio"]

[[servers]]
name    = "tasks"
command = "tasks-mcp"
args    = ["serve", "--mode", "stdio"]
```

Each `[[servers]]` entry requires a `name` (used in logs) and a `command` (must be on `$PATH`). `args` is optional.

See [docs/mcp-services.md](docs/mcp-services.md) for the full server list and configuration reference.

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
- Widget controls include:
	- **New**: start a fresh conversation.
	- **Debug**: show/hide low-level tool execution status lines.
	- **Clear**: clear the visible transcript without deleting conversation history.

Notes:

- Both chat widgets include a **service selector** (Production/Development) and call the selected D-Bus service at `/org/desktopAssistant/Conversations`. Only visible if dev and prod services are both running.
- Widgets auto-detect whether `org.desktopAssistant.Dev` currently has an owner on the session bus.
- If the dev environment is not running, environment selection is hidden.
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
- [Adding MCP Services](docs/mcp-services.md)
- [MCP Integration internals](docs/mcp-integration.md)
- [Development Guide](docs/development.md)
- [Cloud Providers](docs/cloud-providers.md)

## Built-in Tools

The daemon includes built-in tools that are always available, even without external MCP servers:

- **Knowledge base** (unified storage for preferences, memories, and project context):
	- `builtin_knowledge_base_write` — store or update an entry
	- `builtin_knowledge_base_search` — hybrid vector + full-text search
	- `builtin_knowledge_base_delete` — remove an entry by ID
- **Tool discovery**:
	- `builtin_tool_search` — search for additional tools by description; matched tools are automatically activated for the conversation
- **System context**:
	- `builtin_sys_props` — returns date/time, user, hostname, OS, and directory info

Knowledge base data is stored in PostgreSQL (requires database configuration, see [PostgreSQL setup](#postgresql-setup) above). Tool embeddings enable semantic search over both knowledge entries and registered tool descriptions.

## Notes

- If a cloud connector is selected and its API key is missing, daemon still starts but prompt calls fail at runtime.
- If MCP config is missing, daemon runs with no external tools.
- Daemon LLM settings are read from:
	- `$XDG_CONFIG_HOME/desktop-assistant/daemon.toml`, or
	- `~/.config/desktop-assistant/daemon.toml` if `XDG_CONFIG_HOME` is unset.
- Secret backend default is `auto` (local file store first, then systemd credentials, then keyrings).
- KDE Wallet remains supported via `llm.secret.backend = "kwallet"` in `daemon.toml`.
- Conversations, knowledge base entries, and tool registry data persist in PostgreSQL across daemon restarts.

## License

Desktop Assistant is licensed under **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`).
See the [LICENSE](LICENSE) file for the full text.

There are tons of commercial implementations which do not impose FOSS stipulations. This is the open-source alternative. Let's keep it that way.

