# Adelie AI Platform

An API-first AI platform for Linux desktops, deployable as a single-tenant systemd
user service today and intended to scale to multi-tenant server deployments
(k8s, Knative, Lambda) without forking the code path.

The platform ships the **Adele** assistant persona. This repository is the
daemon and its workspace; the chat clients live in their own repos.

> **Experimental.** Most of this codebase is AI-generated and has not been
> exhaustively reviewed. It's useful in practice, but use accordingly.

## Clients

The chat clients are separate repositories, each with their own install steps:

- [adele-tui](https://github.com/adelie-ai/adele-tui) — terminal UI (`adele`)
- [adele-gtk](https://github.com/adelie-ai/adele-gtk) — GTK4 desktop client
- [adele-kde](https://github.com/adelie-ai/adele-kde) — Plasma 6 widgets + KCM

## What it does today

- **Streaming chat** against `ollama`, `openai`, `anthropic`, and `aws-bedrock`,
  with per-conversation model overrides and per-purpose (chat / background /
  vector) routing. Streams are cancellable end-to-end — cancelling a turn tears
  down the in-flight LLM request immediately.
- **Built-in tools** (always available): hybrid vector + full-text knowledge
  base (`builtin_knowledge_base_*`), semantic tool discovery
  (`builtin_tool_search`), and system context (`builtin_sys_props`).
- **MCP tool integration** over stdio for the heavy lifting. Companion servers
  developed for this platform include `fileio-mcp`, `terminal-mcp`,
  `tasks-mcp`, `timeclock-mcp`, `skills-mcp`, and `calendar-mcp`. See
  [docs/mcp-services.md](docs/mcp-services.md).
- **Client-side tool execution.** Tools marked client-local suspend the
  conversation turn to DB, emit `Event::ClientToolCall` to the chat client,
  and resume on `Command::ClientToolResult`. The turn state machine
  (`crates/storage/migrations/017_turn_state.sql`) is the persistence shape
  Lambda would need.
- **Background tasks.** Long-running work (foreground turns, subagents, future
  standalone agents) is tracked in `BackgroundTaskRegistry`, cancellable,
  log-streaming, and durable across daemon restarts
  (`crates/storage/migrations/018_background_tasks.sql`). Clients subscribe via
  `Command::SubscribeBackgroundTasks` and receive `Event::Task*` deltas; every
  `SendMessage` now returns `SendMessageAck { task_id }`.
- **Dream cycle** for knowledge-base consolidation.
- **PostgreSQL + `pgvector`** for conversation history, knowledge base, and
  tool embeddings.

## Transports and auth

The application layer is transport-agnostic
(`crates/transport-dispatch`). Three transports speak the same API:

- **WebSocket** (`crates/ws-interface`) — primary remote transport; HS256 JWT in
  the auth handshake.
- **UDS** (`crates/uds-interface`) — local-only transport authenticated by kernel
  `SO_PEERCRED`; no token. The kernel-attested peer UID *is* the credential, so
  desktop apps reach their own daemon without managing any secret.
- **D-Bus** (`crates/dbus-bridge`) — a standalone `adelie-dbus-bridge` binary
  that fronts the daemon for legacy session-bus consumers. It owns
  `org.desktopAssistant` and speaks to the daemon over UDS like any other
  client. The daemon serves no D-Bus surface of its own, so a D-Bus consumer
  needs this binary installed and running.

Auth follows the **remote-door** rule (#407): the **WebSocket** door requires a
JWT (HS256, shared via `crates/auth-jwt`, issued by the daemon's own WS `/login`
or an external IdP), while **local transports (UDS, D-Bus) authenticate by kernel
peer-cred** and carry no token. The former standalone `adelie-mint` UDS minter is
retired. Production deployments are expected to use an external IdP (Cognito,
Authentik, Keycloak, …); see
[docs/architecture-evolution.md](docs/architecture-evolution.md).

## Multi-tenant by construction

Every personal-data table carries `user_id`; queries are extracted from the
JWT `sub` claim and scoped at the storage layer. A static audit test rejects
unscoped queries at build time. Single-tenant desktop installs collapse to a
fixed default user; multi-tenant servers run the same daemon with a real IdP
in front of it. See [docs/architecture-evolution.md](docs/architecture-evolution.md)
for the target shape and design rules.

That guarantee comes from the database. A daemon with no database configured
runs on a single JSON file instead, and that file holds no owner and has no
partition, so every reader of it reads every conversation in it. This is safe
for the single-user desktop it is meant for, and unsafe the moment a second
person can reach the daemon.

So the daemon **refuses to start** when the remote WebSocket door is enabled and
no database is configured. The refusal names both ways out: configure
`[database] url` (or `DESKTOP_ASSISTANT_DATABASE_URL`), or turn the door off
with `[transports] ws_enabled = false` (or `DESKTOP_ASSISTANT_WS_ENABLED=false`).
A local install is unaffected, because the remote door is off by default. At
every start the daemon logs which conversation store it selected and whether
that store is scoped per user.

## Requirements

- Rust (stable, edition 2024)
- PostgreSQL with the `pgvector` extension
- Linux session D-Bus, if you want `adelie-dbus-bridge` or desktop notifications
- Cloud provider credentials, if you're not running fully on Ollama
- One or more MCP servers, if you want the assistant to do anything beyond chat

### PostgreSQL setup

```bash
# Debian/Ubuntu
sudo apt install postgresql postgresql-contrib postgresql-16-pgvector
# Fedora
sudo dnf install postgresql-server postgresql-contrib pgvector_16
# Arch
sudo pacman -S postgresql postgresql-libs
yay -S pgvector  # or AUR
```

```sql
CREATE USER desktop_assistant WITH PASSWORD 'your_password_here';
CREATE DATABASE desktop_assistant OWNER desktop_assistant;
\c desktop_assistant
CREATE EXTENSION IF NOT EXISTS vector;
```

```toml
# daemon.toml
[database]
url = "postgres://desktop_assistant:your_password_here@localhost/desktop_assistant"
```

Migrations run automatically on startup. Each one is applied at most once and
recorded in a `schema_migrations` table, and the run is serialized by a Postgres
advisory lock, so restarts are cheap however long the history gets and two
daemons pointed at one database queue rather than race.

## Quick start

```bash
# Build
cargo build --workspace

# Configure a connector — default is OpenAI / gpt-5.4
export OPENAI_API_KEY=your_key_here
# or set llm.connector = "ollama" / "anthropic" / "bedrock" in daemon.toml

# Run
cargo run -p desktop-assistant-daemon
```

Connector credentials default to an `auto` secret backend: local file store
first (`$XDG_DATA_HOME/desktop-assistant/secrets/<connector>_api_key`), then
systemd `LoadCredential=`, then desktop keyrings (`libsecret`/`kwallet`), then
environment variables. KDE Wallet remains supported via
`llm.secret.backend = "kwallet"`.

Bedrock follows the standard AWS credential provider chain (env vars, shared
config, SSO, IAM role). `llm.base_url` is interpreted as a region by default
(e.g. `us-east-1`), or as a full Bedrock runtime endpoint URL.

### MCP servers

Without configured MCP servers the assistant can chat but can't take actions
beyond the built-in tools (knowledge base, tool search, system properties).
Create `~/.config/desktop-assistant/mcp_servers.toml`:

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]

[[servers]]
name    = "terminal"
command = "terminal-mcp"
args    = ["serve", "--mode", "stdio"]
```

See [docs/mcp-services.md](docs/mcp-services.md) for the full list.

## systemd user service

```bash
just install-service        # install unit + D-Bus activation mapping
just backend-enable         # auto-start on login (optional)
just backend-status         # status / restart / logs
```

A parallel dev daemon (separate D-Bus name) is available via `just dev-backend`
or as a dedicated user service (`just install-service-dev`).

If D-Bus calls return "The name is not activatable", re-run
`just install-service` / `just install-service-dev` and reload the user manager
(`systemctl --user daemon-reload`).

## Logging

Every log line goes to **stderr**, never stdout, as plain text. `RUST_LOG` sets
the verbosity through `EnvFilter`; the shipped systemd units and container
images set `RUST_LOG=info`.

```bash
RUST_LOG=info desktop-assistant-daemon
RUST_LOG=info,desktop_assistant_mcp_client=debug desktop-assistant-daemon
```

The level contract is a rule, not a preference:

- **INFO** carries ids, counts, byte sizes, durations, model names and token
  counts. Never content.
- **DEBUG** carries prompts, the assembled context, tool arguments, tool
  results, search queries and extracted facts.

So `RUST_LOG=debug` puts conversation content in the journal on purpose. Raise
one target rather than the whole process when you are debugging one thing.

Each turn is one trace: the turn is the root span, each round of the tool loop
is a child of it, and the provider call and every tool dispatch are children of
the round. One `request_id` - the same value a client sees on its own event
stream - is on the turn span and therefore on every line the turn writes, so a
slow turn is followed to the call that was slow from a single identifier.

Metrics are kept in process and summarized every 10 minutes, with no collector
needed. Export of traces, metrics and logs to an OpenTelemetry collector is
available behind an off-by-default `otel` Cargo feature, configured from the
standard `OTEL_*` environment variables:

```bash
cargo build -p desktop-assistant-daemon --features otel
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.example.com:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  desktop-assistant-daemon
```

A default build resolves no opentelemetry crate at all, so `cargo install`
costs nothing extra. See [Logging and telemetry](docs/logging.md) for the full
variable list, the transport trade-offs and the metrics facade.

## Core commands

```bash
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Packaging

```bash
just package-all-docker     # deb, rpm, flatpak (Docker-friendly)
just package-snap           # run on host with snapd/core24
```

## Documentation

- [Architecture](docs/architecture.md)
- [Architecture evolution](docs/architecture-evolution.md) — target shape, design rules
- [API transport](docs/API_TRANSPORT.md)
- [WebSocket API](docs/WEBSOCKET_API.md)
- [D-Bus API](docs/dbus-api.md)
- [MCP services](docs/mcp-services.md) — adding and configuring MCP servers, tool routing, and where each tool runs
- [Development guide](docs/development.md)
- [Logging and telemetry](docs/logging.md) — `RUST_LOG`, the level contract, OTLP export
- [Cloud providers](docs/cloud-providers.md)

## What's not done yet

The multi-agent / multi-transport foundation is in; two follow-ups are tracked
and unfinished:

- `#129` — real cold-restart resume for suspended turns. Today, foreground
  and subagent tasks are marked `Failed` on restart; standalone agents are
  marked `Failed` pending real resume.
- `#135` — first-class `system_prompt` field on `Conversation`.

## License

GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`). See
[LICENSE](LICENSE).
