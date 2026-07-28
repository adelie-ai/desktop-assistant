# WebSocket API

This document describes the desktop-assistant WebSocket API exposed by the daemon.

## Endpoint

- Path: `/ws`
- Default bind: `127.0.0.1:11339` (set with `DESKTOP_ASSISTANT_WS_BIND`)
- URL example: `ws://127.0.0.1:11339/ws`
- Login path: `/login` (HTTP `POST`, Basic auth)

## Authentication

The WebSocket handshake requires a bearer token:

- Header: `Authorization: Bearer <jwt>`
- Missing or invalid token: HTTP `401 Unauthorized` during handshake

Local clients need no token: the UDS door (which the D-Bus bridge also goes
through) authenticates by kernel peer credentials, and the identity is the
connecting peer's OS user. A JWT is a network-door concern, so `/login` is the
way to get one.

Use `/login` with HTTP Basic auth to mint a bearer JWT:

```http
POST /login HTTP/1.1
Host: daemon.example.com
Authorization: Basic <base64(username:password)>
```

Successful response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "subject": "alice"
}
```

`/login` credential validation modes:
- Local Linux host (non-container): validates against current OS user password
  and uses the current OS username (ignores `DESKTOP_ASSISTANT_WS_LOGIN_USERNAME`).
- Container/remote mode: validates against daemon env credentials
  (`DESKTOP_ASSISTANT_WS_LOGIN_USERNAME`, `DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD`).

`/login` authenticates exactly one account, and the `subject` in the response is
the account it authenticated — the same value carried in the token's `sub` claim.
That subject is the user identity the daemon files the connection's data under:
conversations, knowledge entries and scratchpad notes are all scoped by it. So in
container mode the tenant is whatever `DESKTOP_ASSISTANT_WS_LOGIN_USERNAME` names,
and changing that value moves subsequent writes to a different partition. A
request for any other username is rejected (`401`), never issued a token under a
different name.

## Message Model

All payloads are JSON text frames.

### Client -> Server

Envelope:

```json
{
  "id": "req-123",
  "command": { "ping": {} }
}
```

- `id`: client-generated request correlation ID
- `command`: command variant payload

### Server -> Client

Server frames are one of:

1. Result frame

```json
{
  "result": {
    "id": "req-123",
    "result": { "pong": { "value": "pong" } }
  }
}
```

2. Error frame

```json
{
  "error": {
    "id": "req-123",
    "error": "conversation not found"
  }
}
```

An error frame may also carry an optional `detail` object classifying the
outcome, so a caller can act on it programmatically instead of reading prose:

```json
{
  "error": {
    "id": "req-123",
    "error": "not authorized: 'set_api_key' requires the administrator capability; this connection holds tenant",
    "detail": {
      "code": "not_authorized",
      "description": "not authorized: 'set_api_key' requires the administrator capability; this connection holds tenant",
      "message": "Only a daemon administrator can do that. This connection is a tenant.",
      "retryable": false
    }
  }
}
```

- `code` is the stable identifier to branch on: `not_authorized`,
  `unsupported`, `not_found`, `already_terminal`. Never match the message text.
  A handful of rules-based refusals carry a feature-specific code instead, via
  the same field: `CreateConnection`, `UpdateConnection`, and
  `UpsertMcpServer` refuse a `base_url` or MCP server `url` that fails the
  shared remote-URL policy (#804, #895) with one of `url_malformed`,
  `url_scheme_not_allowed`, `url_insecure_scheme`, or `url_target_blocked` —
  see `docs/mcp-services.md`. An older client that does not recognize a code
  still round-trips it unchanged (below), so this never breaks parsing.
- `description` is for logs; `message` is fit to show a person; `retryable`
  says whether repeating the identical request could plausibly succeed.
- `detail` is **absent** when the daemon cannot classify the failure honestly.
  Read that as "unclassified", not as "not an authorization problem".

Backward compatible by construction: `detail` is an optional *field* on the
existing `error` frame, not a new frame variant, so a client that knows only
`{id, error}` parses it unchanged, and `error` repeats `detail.description`. A
`code` a client does not recognize round-trips instead of failing the parse.

3. Event frame (unsolicited/streaming)

```json
{
  "event": {
    "event": {
      "assistant_delta": {
        "conversation_id": "c1",
        "request_id": "srv-req-abc",
        "chunk": "Hello"
      }
    }
  }
}
```

## Commands

Current command variants:

- `ping`
- `get_status`
- `get_config`
  - The config view carries `restart_required`: what is in the daemon's config
    file that the running process is not acting on, as stable area keys
    (`"database"`, `"embeddings"`, `"persistence"`, `"ws_auth"`, `"tls"`,
    `"profiling"`). Absent or empty means every configured value is live. It
    reports *area names only* and never the configured value, so a pending
    `[ws_auth]` or `[tls]` change is visible without disclosing what it was
    changed to. Treat the set as open and render an unrecognized key verbatim
    rather than dropping it. Additive and backward-compatible: an older daemon
    that omits the field deserializes as empty, and an empty report is omitted
    on the wire.
  - The config view also carries `caller_capability`: `"admin"` or `"tenant"`
    for the connection that asked. Read it at connect time and render the
    sections this connection may not change as unavailable, with the reason,
    rather than letting a write fail on submit. Absent from a daemon that
    predates the authorization tier, which means "not reported" - keep the
    prior behaviour then, not "tenant". See "Authorization" below.
  - One key is not an area: `"config_load_failed"` means `daemon.toml` itself
    would not load, so nothing in it is in force and the daemon is running
    built-in defaults. A settings UI should show that state instead of
    presenting the defaults as the user's configuration - an empty connections
    list here means "could not read your config", not "you have none". While it
    is reported, the daemon refuses config-changing commands (`set_config`,
    connection and purpose writes) with an error naming the file, so it cannot
    overwrite a file it could not read. It clears once the file loads again.
- `set_config { changes }`
- `create_conversation { title }`
- `list_conversations { max_age_days }`
- `get_conversation { id }`
- `delete_conversation { id }`
- `clear_all_history`
- `send_message { conversation_id, content }`
- `get_llm_settings`
- `set_llm_settings { connector, model?, base_url? }`
- `set_api_key { api_key }`
- `get_embeddings_settings`
  - The embeddings view (also carried in `get_config`) includes a `health`
    field with the backend's capability-detected state *as of the moment you
    ask*: `{ "status": "disabled" }` (no embedding backend configured, vector
    search off by design), `{ "status": "ok" }` (the backend is producing real
    embeddings), `{ "status": "unavailable", "reason": "..." }` (a backend is
    configured but cannot embed right now, so vector search has degraded to
    full-text search), or `{ "status": "unknown" }` (health was not determined —
    configured but not probed). A startup probe seeds the value; from then on it
    tracks live embed outcomes and a background re-probe that runs while the
    backend is `unavailable`, so a backend that recovers flips back to `ok`
    without a daemon restart and a backend that breaks flips to `unavailable`
    without one either. Changing the `[embeddings]` configuration itself still
    requires a restart. The legacy `available` boolean remains a shallow
    connector check; `health` is the honest signal. Additive and
    backward-compatible:
    `health` defaults to `unknown`, so an older daemon that omits the field
    deserializes as `unknown` (not `disabled` — a working-but-unreported backend
    must not read as off), and a future `status` an older client does not
    recognize also deserializes as `unknown` rather than failing the payload.
- `set_embeddings_settings { connector?, model?, base_url? }`
- `get_connector_defaults { connector }`
- `get_persistence_settings`
- `set_persistence_settings { enabled, remote_url?, remote_name?, push_on_update }`
- `get_database_settings`
- `set_database_settings { url, max_connections }`

Result payloads are typed variants (`pong`, `status`, `conversation_id`, `conversations`, `conversation`, `config`, `ack`, etc.).

### Credentials in connection URLs

Every connection URL a reply carries — the database `url`, the git
`remote_url` in `get_persistence_settings` and `get_config` — has its password
replaced by `***`:

```json
{"result": {"database_settings": {"url": "postgres://adele:***@postgres:5432/adele", "max_connections": 5}}}
```

The rest of the URL (scheme, user, host, port, database, options) is intact, so
a settings UI can still show what is configured. Both the inline
`user:password@` form and the libpq `?password=` parameter are redacted.

Writes round-trip. Posting back a URL that still carries the `***` placeholder
keeps the stored password, provided nothing else in the URL changed — so a form
that only ever saw the redacted value can still save an unrelated field.
Editing any other part of the URL means re-entering the password: the daemon
refuses such a write rather than splicing its stored credential into a URL the
client has edited, and it never stores the placeholder as if it were a
password.

## Events

Current event variants:

- `config_changed { config }`
- `assistant_delta { conversation_id, request_id, chunk }`
- `assistant_completed { conversation_id, request_id, full_response }`
- `assistant_error { conversation_id, request_id, error }`
- `assistant_status { conversation_id, request_id, message, capability_change? }`

### Tool-provenance gating (behaviour change)

A turn that has taken in content from outside the trust boundary - a fetched
web page, a third-party API result, a file read at a path the model chose -
refuses the tools that could act on it for the remainder of that turn. The
closed tiers are `mutate`, `network_egress`, `code_execution`, and anything the
daemon cannot classify. Reading, and output to the user's own session, stay
open. The next turn starts clean.

**What an integrator must do.** A model-chosen tool call that would previously
have run may now come back refused. The refusal arrives as an ordinary tool
result, not an error, and the turn keeps going - so a client that does nothing
still works and simply sees the assistant take another path. To handle it
deliberately:

1. Watch `assistant_status` for `capability_change`. It is present only on the
   one status per turn that reports the narrowing, and absent on ordinary
   progress:

   ```json
   {"assistant_status": {
     "conversation_id": "c1",
     "request_id": "r1",
     "message": "Read outside content - sending, changing and running are off for the rest of this turn",
     "capability_change": {
       "reason": "external_content_ingested",
       "closed_tool_tiers": ["mutate", "network_egress", "code_execution", "unclassified"]
     }
   }}
   ```

   The change holds for the turn named by that `request_id`.

2. Read `tool_tier` on `ToolUsageView` to know in advance which of the tools a
   conversation uses can be refused. Tier values: `read`, `present`, `mutate`,
   `network_egress`, `code_execution`, `unclassified`. The last four are the
   ones that close.

3. To get a refused action done, start a new turn that does not read outside
   content first.

Both fields are optional additions. A client built before them keeps parsing
every frame unchanged, and a tier or reason string it does not recognise must
be treated as `unclassified` / `unknown` - which is to say, assume it can be
refused.

## Typical Session Flow

1. Acquire JWT (local clients)
- Call D-Bus `GenerateWsJwt("my-client")` (token subject is current OS username).

2. Acquire JWT (remote clients, no D-Bus)
- `POST /login` with Basic auth.
- Receive `token`.

3. Open WebSocket
- Connect to `ws://127.0.0.1:11339/ws`.
- Include `Authorization: Bearer <token>`.

4. Health check
- Send `ping`.
- Expect `result -> pong`.

5. Discover or create a conversation
- Send `list_conversations`.
- If needed, send `create_conversation`.

6. Send a user message
- Send `send_message`.
- First response is `result -> ack`.
- Then receive streamed events:
  - one or more `assistant_delta`
  - terminal `assistant_completed` (or `assistant_error`)

7. Refresh conversation state
- Send `get_conversation` if you need the full canonical message list.

8. Optional live configuration
- Send `set_config`.
- Expect:
  - `result -> config`
  - followed by `event -> config_changed` with the same config snapshot.
- `set_ws_auth_settings` still answers with a bare `ack`, and is followed by an
  `event -> config_changed`. Its `restart_required` is how the client that made
  the change learns that the new authentication methods, OIDC config, or allowed
  origins are written but not yet in force.

## Authorization

Authenticating a token says who you are. It does not say that you own the
service. A WebSocket connection is a **tenant** unless `[authz] admin_subjects`
in `daemon.toml` names its subject (the JWT `sub`), in which case it is an
**administrator**. The allowlist is empty by default, and **no command writes
it**: it is set by editing `daemon.toml`, so the API surface offers a tenant no
way to grant themselves the capability. (Daemon-side file and shell MCP servers
run as the daemon's own uid and could reach the file; they ship disabled, and
constraining them belongs to the tool-execution design, not this tier.)

A tenant may do everything with its own data: conversations, messages,
knowledge, scratchpads, background tasks, personality, and every read on this
API, including `list_connections`, `list_available_models` and `get_purposes`,
which the model picker needs.

An administrator is additionally required for the service-configuration writes:

`set_api_key`, `set_embeddings_settings`, `set_persistence_settings`,
`set_database_settings`, `set_backend_tasks_settings`, `set_ws_auth_settings`,
`create_connection`, `update_connection`, `delete_connection`,
`set_connection_secret`, `set_purpose`, `add_mcp_server`, `remove_mcp_server`,
`set_mcp_server_enabled`, `upsert_mcp_server`, `set_mcp_secret`,
`upsert_service_account`, `remove_service_account`,
`start_knowledge_maintenance`, and `mcp_server_action` for every verb except
`"status"`.

`set_config` is decided by its payload, and every field it carries writes
daemon-global state, so any non-empty change set needs the administrator
capability - the personality traits included, because they are one global block
that every conversation without an override resolves against. A tenant sets
their own disposition with `set_conversation_personality`.

`mcp_server_action` is split by its verb: `"status"` is a read and stays tenant;
`"start"`, `"stop"`, `"restart"` and anything added later need the
administrator capability.

`start_knowledge_maintenance` needs the administrator capability: every
operation reaches past the caller's own rows, up to nulling and re-deriving
every tenant's embeddings through the operator's provider.

Discover your own level from `caller_capability` on the `get_config` reply (and
on every `config_changed` event) rather than probing with a write.

A refused command returns the error frame shown above with
`detail.code = "not_authorized"` and `retryable: false`. The connection stays
open, and the command has no effect - the check runs before the handler.

**This is a behaviour change for existing API consumers.** A remote caller that
could previously issue any of the commands above will now be refused unless it
is listed. If your integration configures the daemon, add its subject to
`[authz] admin_subjects` and restart the daemon. Integrations that only send
messages and read their own data are unaffected. So are local clients on the
daemon's own account, which are administrators by construction over the Unix
socket and D-Bus.

## Notes

- The command `id` correlates only `result`/`error` frames.
- Streaming assistant events are correlated by server-generated `request_id` inside event payloads.
- Multiple requests can be in flight concurrently; clients should match by `id` and event metadata.
