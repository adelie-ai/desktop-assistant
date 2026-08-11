# Adding MCP Services

## Why MCP Services Matter

The Adelie platform uses [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/) as its primary mechanism for giving the LLM access to tools — file I/O, web search, calendar access, system control, and so on. **Without at least one MCP server configured, the assistant has very limited ability to take actions on your behalf.**

The built-in tools (preference memory, factual memory) are always available, but real-world usefulness depends heavily on external MCP servers providing capabilities relevant to your workflow.

These MCP servers are not the full extent of Adelie platform's functionality. It is usually capable of working out very complex tasks for which it has not been explicitly programmed. You should think of these MCP servers as the building blocks it uses to synthesize more complex behaviors. By providing deterministic abstractions over complex behaviors in this way, the service doesn't need to think so hard and can worry about other things and get to your end result more quickly. 

## Available MCP Servers

The following MCP servers (not an exhaustive list) are developed alongside the Adelie platform and are designed to work with it out of the box.

### fileio-mcp

File system operations for LLM agents: read, write, structured edit, line-aware read, file/content search, copy, move, stat, mkdir, remove, symlinks, permissions, and more.

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]
```

### terminal-mcp

Shell execution for LLM agents. Exposes `terminal_execute` plus a dynamic script tool lifecycle (`terminal_store_script`, `terminal_remove_script`, `terminal_list_scripts`, and per-script `script_<name>` tools). Results include `exit_code`, `stdout`/`stderr`, timeout status, and truncation flags.

```toml
[[servers]]
name    = "terminal"
command = "terminal-mcp"
args    = ["serve", "--mode", "stdio"]
```

> **Note:** Terminal execution is a high-privilege capability. Audit logging is available via `MCP_TERMINAL_LOG_DIR`.

### tasks-mcp

Local task management backed by Markdown files with YAML frontmatter. Supports multiple lists/contexts, a simple `epic → deliverable` hierarchy, and atomic file writes. Storage defaults to `~/.local/share/desktop-assistant/tasks/`.

```toml
[[servers]]
name    = "tasks"
command = "tasks-mcp"
args    = ["serve", "--mode", "stdio"]
```

### timeclock-mcp

Time tracking for projects. Tools: `timeclock_project_upsert`, `timeclock_project_list`, `timeclock_project_delete`, `timeclock_clock_in`, `timeclock_clock_out`, `timeclock_session_get_active`, `timeclock_session_query`.

```toml
[[servers]]
name    = "timeclock"
command = "timeclock-mcp"
args    = ["serve", "--mode", "stdio"]
```


## Configuration File

MCP servers are configured in:

```
$XDG_CONFIG_HOME/desktop-assistant/mcp_servers.toml
```

Which typically resolves to:

```
~/.config/desktop-assistant/mcp_servers.toml
```

Create this file if it does not exist. The daemon reads it at startup; restart the daemon after any changes.

## File Format

Each MCP server is declared as a `[[servers]]` entry:

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]
```

Fields:

| Field       | Required | Description                                                                          |
|-------------|----------|--------------------------------------------------------------------------------------|
| `name`      | yes      | Logical label for this server; used in logs and startup diagnostics                  |
| `command`   | for stdio | Executable to spawn — must be on `$PATH` or an absolute path. Omit when using `[servers.http]` |
| `args`      | no       | Command-line arguments passed to the process (default: empty list)                   |
| `namespace` | no       | If set, all tools from this server are exposed as `{namespace}__{tool_name}`; if absent, tool names are passed through unchanged |
| `env`       | no       | Extra environment variables for the process, as `[servers.env]` key/value pairs |
| `env_secrets` | no     | Environment variables whose value is looked up by ID from `secrets.toml`, as `[servers.env_secrets]` key/secret-id pairs |
| `inherit_env` | no     | Names of variables this server is opted in to receive from the daemon's own environment, beyond the [always-passed-through allowlist](#environment-variables) — see [Per-server opt-in](#per-server-opt-in-inherit_env) |
| `[servers.http]` | no  | Reach the server over HTTP instead of spawning `command` — see [Remote (HTTP) MCP Servers](#remote-http-mcp-servers) |

The daemon communicates with each server over stdio using the MCP JSON-RPC protocol.

## Environment Variables

A spawned stdio server does not inherit its parent's environment. This is
deliberate: the parent's environment can hold values a server has no reason
to see, such as the database connection string. The rule is the same for
both places that spawn a stdio server: the daemon's own fleet (this page) and
the [client-side MCP host](client-mcp-host.md), which runs on a real desktop
session where D-Bus and audio genuinely exist.

The daemon passes through only a small, named set of variables from its own
environment:

| Variable | Why |
|----------|-----|
| `PATH` | Resolve the server's own subprocess dependencies (a bundled browser, shell tools, and so on) |
| `HOME` | Config/cache directory fallback |
| `USER`, `TMPDIR`, `TERM` | `terminal-mcp` reads exactly `PATH, HOME, USER, TMPDIR, TERM, LANG` from its own process environment as part of its defence-in-depth env scrub before running a command |
| `LANG` | Locale-dependent output formatting |
| `TZ` | Local-time timestamps |
| `HTTP_PROXY`, `http_proxy`, `HTTPS_PROXY`, `https_proxy`, `NO_PROXY`, `no_proxy` | Outbound HTTP through a proxy |
| `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_STATE_HOME` | Standard config/data/cache/state directories |
| `WEB_CHROME_PATH` | Path to a bundled Chromium binary (`web-mcp`) |
| `SKILLS_MCP_ROOTS` | Skill-root search path (`skills-mcp`, enabled by default) |
| `SKILLS_MCP_WRITE_ROOT` | Where `skills-mcp` (enabled by default) writes a new skill when the default root is not writable |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_TIMEOUT`, and the `_TRACES_`, `_METRICS_` and `_LOGS_` form of each | Where a server exports its own traces, metrics and log records, and how. A server that receives none of these exports nothing — see [Logging and telemetry](logging.md) |
| `OTEL_RESOURCE_ATTRIBUTES` | Deployment context (pod, namespace, node) on a server's own signals, so they read beside the daemon's |
| `RUST_LOG` | Log filter. One filter governs a server's console output and its exported log records together, the same as for the daemon |

`OTEL_EXPORTER_OTLP_HEADERS` is **not** passed, and neither are its
per-signal forms. That variable carries the backend ingestion credential, and
every spawned server — including a third-party one you add — would receive
it. Servers export to the collector, and the collector holds the backend
credential. A server that must reach a backend directly takes the scoped
route below, `inherit_env`.

`OTEL_SERVICE_NAME` is not passed either: every server would report under the
daemon's service name. Each server names itself.

Every other variable is stripped, even one the daemon itself received. A
server that needs something else must receive it through its own `env` (or
`env_secrets`, for a value stored in `secrets.toml`) in its `[[servers]]`
entry:

```toml
[[servers]]
name    = "my-server"
command = "my-server-mcp"
args    = ["serve"]

[servers.env]
MY_SERVER_SETTING = "value"
```

A server's own `env`/`env_secrets` always wins over a passed-through value of
the same name.

**This is a behaviour change.** Before this list existed, a spawned server
inherited the daemon's whole environment. A server that read an inherited
variable not on this list will stop seeing it — set that variable in the
server's own `env` instead. If a server genuinely needs a variable passed
through globally rather than per-server, open an issue on
`adelie-ai/desktop-assistant` naming the server and the variable; the list is
kept deliberately short and evidence-based (see `ENV_PASSTHROUGH_ALLOWLIST`
in `crates/mcp-client/src/lib.rs`), not grown on request.

### Per-server opt-in: `inherit_env`

A few variables are genuinely useful to exactly one shipped server, but would
be a bad idea to grant every spawned server — including a third-party one an
operator adds — by default. `DBUS_SESSION_BUS_ADDRESS` and `XDG_RUNTIME_DIR`
are the concrete case: both are also exactly what a stock D-Bus client
library uses to auto-discover the session bus, which fronts the freedesktop
Secret Service holding connector API keys and MCP OAuth tokens. Granting
either one globally would give every spawned stdio server a route to that
credential store by default — a real lowering of the security bar, not a
theoretical one, even though the same uid could in principle reconstruct the
bus address by other means.

`inherit_env` names the variables a *specific* server is opted in to receive
from the daemon's own environment, on top of `env`/`env_secrets` (which still
win on a name collision):

```toml
[[servers]]
name        = "tasks"
command     = "tasks-mcp"
args        = ["serve"]
inherit_env = ["DBUS_SESSION_BUS_ADDRESS"]
```

The shipped default config (`deploy/mcp/mcp_servers.default.toml`) sets this
for exactly two servers: `tasks` (`DBUS_SESSION_BUS_ADDRESS`, for its
session-bus signal service that refreshes QML widgets) and `internet-radio`
(`XDG_RUNTIME_DIR`, so the `mpv` it spawns can find the PipeWire/PulseAudio
session socket). Every other server — including any third-party one you
add — does not receive either variable unless you opt it in explicitly here.

### Upgrading an existing install

The daemon seeds the shipped default config on **first boot only** — it never
overwrites an `mcp_servers.toml` that already exists (`ensure_mcp_config_exists`
in `crates/mcp-client/src/config.rs`). The `inherit_env` grants above therefore
reach a fresh container install automatically, but **not** an existing one:
your own `mcp_servers.toml` (daemon) or `client-mcp.toml` (client-side host,
see [Client-side MCP host](client-mcp-host.md)) is untouched. This matters
most on a real desktop session — exactly where `DBUS_SESSION_BUS_ADDRESS` and
`XDG_RUNTIME_DIR` mean something and where the credential store they front
actually lives — since a headless fleet container is the case that seeds
fresh most often.

If you already have a `tasks` or `internet-radio` entry (or an
equivalent third-party server that needs its own session-bus/runtime-dir
variable) from before this change, add the relevant `inherit_env` line to it
by hand:

```toml
[[servers]]
name        = "tasks"
command     = "tasks-mcp"
args        = ["serve"]
inherit_env = ["DBUS_SESSION_BUS_ADDRESS"]

[[servers]]
name        = "internet-radio"
command     = "internet-radio-mcp"
args        = ["serve"]
inherit_env = ["XDG_RUNTIME_DIR"]
```

Without it, `tasks`' QML-widget-refresh signal service silently stops working
(it treats a bus failure as non-fatal) and `internet-radio`'s spawned `mpv`
cannot find the audio session — no error, just a feature that quietly does
less than it used to.

## Tool Namespacing

By default, tool names are passed through exactly as the MCP server reports them. Set the optional `namespace` field to prefix all tools from that server:

```
{namespace}__{tool_name}
```

For example:

```toml
[[servers]]
name      = "fileio"
command   = "fileio-mcp"
args      = ["serve", "--mode", "stdio"]
namespace = "fs"
```

This exposes `fileio-mcp`'s `fileio_read_file` as `fs__fileio_read_file`.

**When to use namespacing:**

- **Collision avoidance** — multiple servers that expose tools with the same name (for example, `open_ticket` from a built-in tasks server, Jira, and Bugzilla):

```toml
[[servers]]
name      = "tasks-builtin"
command   = "tasks-mcp"
namespace = "tasks"

[[servers]]
name      = "jira"
command   = "jira-mcp"
namespace = "jira"

[[servers]]
name      = "bugzilla"
command   = "bugzilla-mcp"
namespace = "bz"
```

This exposes `tasks__open_ticket`, `jira__open_ticket`, and `bz__open_ticket` as distinct tools.

- **Multiple instances of the same server** — two `fileio-mcp` processes scoped to different directories:

```toml
[[servers]]
name      = "work-files"
command   = "fileio-mcp"
args      = ["--root", "/home/user/work"]
namespace = "work"

[[servers]]
name      = "personal-files"
command   = "fileio-mcp"
args      = ["--root", "/home/user/personal"]
namespace = "personal"
```

This exposes `work__fileio_read_file` and `personal__fileio_read_file` as distinct tools.

When `namespace` is absent, tool names are forwarded to the LLM exactly as reported by the server — suitable for servers that already use unique, self-describing names (`fileio_read_file`, `terminal_execute`, etc.).

## Multiple Servers

Add as many `[[servers]]` blocks as needed:

```toml
[[servers]]
name    = "fileio"
command = "fileio-mcp"
args    = ["serve", "--mode", "stdio"]

[[servers]]
name    = "websearch"
command = "websearch-mcp"
args    = []

[[servers]]
name    = "calendar"
command = "/opt/my-mcp-servers/calendar-mcp"
args    = ["--profile", "work"]
```

## Remote (HTTP) MCP Servers

Besides spawning a local process over stdio, the daemon can reach a **remote** MCP server over HTTP (the MCP *streamable-HTTP* transport). Add a `[servers.http]` table instead of a `command`:

```toml
[[servers]]
name      = "gmail-personal"
namespace = "gmail_personal"

[servers.http]
url                = "https://gmailmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_personal_token"
```

Fields under `[servers.http]`:

| Field                | Required | Description                                                                                |
|----------------------|----------|--------------------------------------------------------------------------------------------|
| `url`                | yes      | Remote MCP endpoint. Its presence selects the HTTP transport. See "Which URLs are accepted" below |
| `auth_bearer_secret` | no       | Secret **ID** (looked up in `secrets.toml`) whose value is sent as a **static** `Authorization: Bearer` token — never refreshed |
| `[servers.http.oauth]` | no     | Authenticate with OAuth 2.0 instead: the daemon refreshes short-lived access tokens on its own — see [OAuth 2.0](#oauth-20-google) |

### Which URLs are accepted

Every request to `url` carries the bearer token or OAuth access token above,
so `url` is checked against the same rule as a connection's `base_url`
(#804, #895):

- `https://` is required for a host reachable over the open network.
- Plain `http://` is accepted only to loopback (`localhost`, `127.0.0.1`,
  `::1`), a private network address (RFC1918, e.g. `192.168.x.x`), or a bare
  hostname with no dot — the shape of a Kubernetes short Service name, such
  as `http://homeassistant-mcp:8080` reached over an in-cluster network.
  A dotted hostname (`mcp.example.com`) always needs `https://`, even on a
  LAN.
- A link-local address, an unspecified address (`0.0.0.0` / `::`), the cloud
  metadata address `169.254.169.254`, and the GCP metadata hostname
  `metadata.google.internal` are refused regardless of scheme.

The daemon checks this both when the config is saved (`UpsertMcpServer`, so a
bad URL is refused immediately with a clear reason) and when it connects, so
a `mcp_servers.toml` edited by hand gets the same protection. A refusal
names the rule it hit; see `docs/WEBSOCKET_API.md` for the wire shape
(`detail.code`: `url_malformed`, `url_scheme_not_allowed`,
`url_insecure_scheme`, or `url_target_blocked`).

The bearer token itself is never written in `mcp_servers.toml` — only the secret **ID** is. Put the real token in `secrets.toml` (also enforced `0600`):

```toml
# ~/.config/desktop-assistant/secrets.toml
[secrets]
google_personal_token = "ya29.a0Af..."
```

> **Static tokens don't refresh.** A value placed under `auth_bearer_secret` is sent verbatim; if it's a short-lived OAuth access token it will stop working when it expires. For anything that expires, use an [OAuth block](#oauth-20-google) so the daemon refreshes it automatically.

### OAuth 2.0 (Google)

For tokens that expire (Google's do, in ~1 hour), add a `[servers.http.oauth]` table. The daemon then holds a long-lived **refresh token** and exchanges it for fresh access tokens on demand — including an automatic retry when the server answers `401`.

```toml
[[servers]]
name = "gmail-work"
namespace = "gmail_work"

[servers.http]
url = "https://gmailmcp.googleapis.com/mcp/v1"

[servers.http.oauth]
client_id         = "1234567890-abc.apps.googleusercontent.com"
token_url         = "https://oauth2.googleapis.com/token"
authorize_url     = "https://accounts.google.com/o/oauth2/v2/auth"
refresh_token_ref = "gmail_work_refresh"     # secret ID in secrets.toml
client_secret_ref = "google_client_secret"   # secret ID; omit for public/PKCE clients
account           = "dave@example.com"        # token-store key; share across services for one account
scopes = [
  "https://www.googleapis.com/auth/gmail.modify",
  "https://www.googleapis.com/auth/calendar",
]
```

Fields under `[servers.http.oauth]`:

| Field                 | Required | Description                                                                       |
|-----------------------|----------|-----------------------------------------------------------------------------------|
| `client_id`           | yes      | OAuth client identifier (public; safe to store inline)                            |
| `token_url`           | yes      | Token endpoint (HTTPS), e.g. `https://oauth2.googleapis.com/token`                 |
| `refresh_token_ref`   | yes      | Secret **ID** holding the refresh token (minted by the login command below)       |
| `client_secret_ref`   | no       | Secret **ID** for the client secret; omit for public (PKCE-only) clients          |
| `authorize_url`       | for login | Authorization endpoint; only used by the interactive login                       |
| `scopes`              | for login | Scopes requested during login (they determine which tools/writes are permitted)  |
| `account`             | no       | Token-store key (defaults to the server `name`)                                   |
| `refresh_skew_seconds`| no       | Refresh this many seconds before hard expiry (default `60`)                        |

**One-time login.** Run the interactive flow once per account to mint the refresh token. It opens your browser (installed-app loopback + PKCE), captures the redirect on `127.0.0.1`, and writes the refresh token into `secrets.toml` under `refresh_token_ref`:

```bash
desktop-assistant-daemon --mcp-oauth-login gmail-work
# → opens browser, then: "Saved refresh token for 'gmail-work' … Restart the daemon."
```

Then restart the daemon; it will keep the access token fresh from there on. Secret **values** (client secret, refresh token) live only in `secrets.toml` (`0600`) — never in `mcp_servers.toml`.

At runtime the daemon caches the live token in the system secret store (best-effort) so a restart needn't re-fetch it; where that cache lives is an internal detail and needs no configuration — with no secret store available (headless) the daemon just keeps the token in memory. If you re-run the login (new refresh token), the daemon detects the change on next start and re-bootstraps automatically.

> **Workspace domains skip the weekly re-auth.** If your `token_url` account is on a Google Workspace domain you control, set the OAuth **consent screen to "Internal"** — the refresh token then does not expire after 7 days and needs no Google verification, even for restricted scopes like `gmail.modify`. Personal/"Testing" consent screens expire the refresh token weekly; `--mcp-oauth-login` will need re-running when that happens (the daemon logs an `invalid_grant` error telling you so).

### Google Workspace (Gmail / Calendar / Drive / Chat)

Google hosts a first-party MCP endpoint per Workspace service; each is one `[[servers]]` entry:

```toml
[[servers]]
name = "gmail"
namespace = "gmail"
[servers.http]
url = "https://gmailmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_token"

[[servers]]
name = "calendar"
namespace = "calendar"
[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_token"
```

(Which tools — and whether writes like sending mail or RSVPing invites are permitted — depends on the OAuth scopes granted to your token.)

**Multiple accounts.** Give each account its own entry with a distinct `namespace` and `auth_bearer_secret`, so the assistant can tell them apart ("create an invite on my *work* calendar" → the `calendar_work__` tools):

```toml
[[servers]]
name = "calendar-personal"
namespace = "calendar_personal"
[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_personal_token"

[[servers]]
name = "calendar-work"
namespace = "calendar_work"
[servers.http]
url = "https://calendarmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_work_token"
```

The examples above use a static `auth_bearer_secret` for brevity; in practice each account entry should carry its own [`[servers.http.oauth]`](#oauth-20-google) block instead (with a distinct `refresh_token_ref` and `account`), so the daemon keeps every account's token fresh on its own. Two services for the *same* account (e.g. Gmail + Calendar) can share one `account` key so a single login covers both.

Within a single account, choosing between that account's calendars (primary vs. a shared "XYZ" calendar) is handled by the server's own `calendarId` tool argument, not by configuration.

A full end-to-end walkthrough for Google's endpoints — creating the OAuth client, the scope list, and the one-time sign-in — is in [Google Workspace setup](RemoteMCP/GoogleWorkspace-setup.md).

## What Tool Discovery Tells the Model

`builtin_tool_search` reports where each hit runs, because a tool's name and
description do not say which machine it acts on. Each result carries a
`runs_on` value:

| `runs_on` | What it means |
|---|---|
| `daemon` | A built-in, or an MCP server the daemon spawned. Acts on the daemon's own files and processes. |
| `remote-service` | An MCP server the daemon reaches over HTTP. Acts on that service, and on no local files. |
| `device` | A tool the connected client registered. Acts on the user's own machine. |

Each hit also carries `host`, the name of the machine that issues the call: the
daemon's own label for a `daemon` or `remote-service` hit, and the connected
client's label for a `device` hit (`your device` when the client reported none).
The two fields answer different questions. `runs_on` says what a tool reaches,
which is what stops a model believing a remote calendar tool can read local
files. `host` says which machine issues it, which is what a request about a
particular machine - "read that file on the laptop" - is matched against.

The daemon and remote-service split is read live from the routing table and the
server configuration, so a server added since startup classifies correctly. A
name the executor does not route is a built-in, which runs inside the daemon
process, so it reports `daemon`.

Client-registered tools are searched too. They are registered per connection and
never written to the tool registry, so a search that consulted only the registry
could never offer the option that acts on the user's own machine. They are
matched lexically against the query - the set is tens of tools, and no embedding
exists for it.

Each response also carries `same_machine` and a one-line `runs_on` legend naming
the daemon's machine. Only the runner values present in the results are
described. When the daemon and the client are the same machine, a client tool
and a daemon tool of the same name are the same capability, so the daemon entry
is kept and the duplicate is dropped - matching how the turn loop resolves that
collision.

A search that matched more client tools than it returned reports the count it
dropped in `more_device_tools_matched`.

## One Name, Two Machines

A tool is identified by the pair (capability name, host). The name says what the
tool does and never says where it runs, so a machine that is renamed or replaced
changes no tool name - and what the assistant learned about a tool on one
machine still applies to the same tool on another.

Each turn builds one table of every tool it can reach, keyed by that pair. The
table decides both what the model is shown and where each call goes, so the
schema the model reads and the machine that runs the call are one decision
rather than two that can disagree.

When the daemon and the connected client both offer a capability under one name:

- The advertised schema gains an optional `__host` argument, whose values are
  the `runs_on` tokens of the hosts that offer it (`daemon`, `device`). The
  model sets it when the task is about a particular machine.
- With no `__host` stated, the routing policy chooses. The policy is
  `prefer-co-located`: the call stays on the daemon, which is the process
  running the turn, rather than crossing a socket to a client that may
  disconnect mid-turn.
- `__host` is the harness's field, not the tool's. It is removed before the tool
  runs, so no tool receives it.
- A capability only one host offers carries no `__host` argument: there is
  nothing to choose.

An operator who wants a client-registered tool to be the default for a name the
daemon also holds should give it a name of its own - a namespace on the client's
MCP entry does that. The daemon logs the shared names once per turn.

## Startup Behaviour

When the daemon starts:

1. Each configured server process is spawned.
2. The daemon performs the MCP `initialize` handshake.
3. `tools/list`, `resources/list`, and `prompts/list` are fetched from each server.
4. A routing table is built mapping tool names → server index.

If a server fails to start, a warning is logged and the daemon continues without that server's tools. No server failure is fatal to the daemon.

If the server process exits before completing the handshake, the logged error names the exit status **and quotes what the server last wrote to stderr**, so a rejected command line, a missing file, or a refused credential is diagnosable from the log line instead of reading as a generic protocol failure:

```
MCP server exited with status 2 before completing the handshake; it last wrote this to stderr: error: the following required arguments were not provided: --config <CONFIG>
```

A server does not have to exit to fail, and the other three shapes carry the same clause. A server that stops answering is reported as a timeout, so a hang — the startup failure that is hardest to read, because nothing exited and nothing was refused — still names its cause:

```
MCP request 'initialize' timed out after 30s of silence; it last wrote this to stderr: fatal: cannot open database
```

A server that abandons its stdout but keeps running has no exit status to name, and reports what it said instead:

```
MCP server closed stdout; it last wrote this to stderr: fatal: no write access to the state directory
```

A server that has already gone when the daemon next writes to it fails on the write, because the read end of its stdin pipe is closed. The error keeps its I/O class and gains the same account, so which side of the exchange noticed first is not something a reader has to work out:

```
I/O error communicating with MCP server: Broken pipe (os error 32); MCP server exited with status 4 before completing the handshake; it last wrote this to stderr: fatal: lost its database connection
```

The quoted tail is bounded: the last 10 completed lines, plus an unterminated final fragment where the server left one, each capped at 512 bytes and marked with `...` where it was cut, joined with ` | ` onto one line. The fragment matters because a server killed part-way through a write, or one that prints its complaint without a trailing newline, puts its most useful line there and never terminates it. Lines beyond that are discarded as they arrive, so a server that floods stderr cannot enlarge the message. Characters that would re-shape the line the message lands in are replaced with spaces before it is quoted — the C0/C1 controls, the Unicode format characters (bidi overrides, zero-width marks) and U+2028/U+2029, which a JSON field and an HTML renderer both treat as a line break. The text is a server-chosen remote string, and a renderer must still escape it for its own medium.

A server's stderr appears **only** in this failure message. It is never streamed to the log as it arrives, at any level, because it is the server's own unfiltered output and can carry a credential or a fragment of user content — see [Logging](logging.md#what-may-appear-at-each-level).

Where the server exits without writing anything, there is no evidence to quote, and the message instead suggests the most common silent cause — an environment variable the server needed that is not on the [pass-through allowlist](#environment-variables) and not in its own `env`:

```
MCP server exited with status 7 before completing the handshake and wrote nothing to stderr; if it needs an environment variable, set it in this server's own `env` config (see docs/mcp-services.md#environment-variables) rather than relying on it being inherited
```

The same message reaches the settings/KCM panel's per-server detail field, so an operator sees it without reading the log.

## Verifying Loaded Tools

Check daemon logs to confirm servers and tools were loaded:

```bash
just backend-logs
# or, for the dev daemon:
just backend-dev-logs
```

Look for lines like:

```
INFO connecting to MCP server 'fileio': fileio-mcp
INFO MCP server 'fileio' provides 8 tools
```

If a server failed to connect you will see:

```
ERROR failed to connect to MCP server 'fileio': ...
```

## Applying Changes

After editing `mcp_servers.toml`, restart the daemon to reload:

```bash
just backend-restart
# or for the dev daemon:
just backend-dev-restart
```

## Further Reading

- [Google Workspace setup](RemoteMCP/GoogleWorkspace-setup.md) — end-to-end OAuth walkthrough for Google's remote MCP endpoints
- [D-Bus API](dbus-api.md) — how clients invoke tools via the conversation API
