# API + Transport Strategy (D-Bus + WebSocket)

## Motivation
`desktop-assistant` is Linux-desktop-first (D-Bus), but users also need to connect remotely (e.g. macOS client while the daemon runs on a Linux host accessed over SSH).

To avoid maintaining two divergent APIs, **D-Bus and WebSocket must expose the same API surface and semantics**.

## Design: common handlers, thin adapters
Implement protocol-neutral application handlers and shared API types once, then provide adapters:

- **Shared API model** (serde types): canonical `Command` / `Result` / `Event` structures.
- **Application layer**: validates commands, performs actions, emits canonical events.
- **D-Bus adapter**: maps D-Bus methods + signals to/from canonical commands/events.
- **WebSocket adapter**: maps WS request/reply + event stream to/from canonical commands/events.

This mirrors the Ports & Adapters approach in `AGENTS.md`.

## v1 API surface (keep small)

### Commands
- `Ping`
- `GetStatus`
- `SendMessage { conversation_id?, content }` (streaming response)
- `GetConfig`
- `SetConfig { changes }`

### Events
- `StatusChanged(Status)`
- `ConfigChanged { config }`
- `MessageStarted { message_id, role }` (optional)
- `AssistantDelta { request_id, chunk }`
- `AssistantCompleted { request_id, full_response }`
- `Error { code, message, retryable }`

## Config
Settings are expected to be small in number (~10). Prefer a **typed config struct** for v1:
- `GetConfig -> Config`
- `SetConfig(ConfigChanges) -> Config`

`Config` also reports `restart_required`: what is on disk that the running
process is not acting on. Some subsystems (TLS, WS auth, the embedding client)
are wired once at startup; a reload applies every hot knob in the same edit and
reports the rest here rather than accepting the change silently. Area names
only, never configured values.

The key `config_load_failed` is the whole-file case: `daemon.toml` would not
load, so the daemon is running built-in defaults and refuses config-changing
commands until the file loads again - it will not overwrite a file it could not
read.

If settings later grow significantly, introduce `ListConfigSchema` and move to a registry-based key/value system.

## Authorization: tenant and administrator

Authentication says which subject a connection is. Authorization says what that
subject may do, and there are two levels.

- **Tenant** - the caller's own conversations, knowledge, scratchpads,
  background tasks and preferences.
- **Administrator** - additionally the service itself: provider credentials,
  connectors and purposes, the database, the WebSocket auth posture, and which
  child processes the daemon spawns for MCP.

The daemon grants the administrator capability two ways:

1. **A local peer whose kernel-attested uid equals the daemon's own.** On a Unix
   socket `SO_PEERCRED` is not forgeable, so the account that runs the daemon
   administers it. A single-user desktop therefore needs no configuration.
2. **A subject named in `[authz] admin_subjects` in `daemon.toml`.** Empty by
   default. This is the only way a remote (WebSocket) caller becomes an
   administrator. The section is file-only: no command writes it, so a tenant
   cannot grant themselves the capability they are being denied. Editing it
   needs a daemon restart, and a reload reports `authz` in `restart_required`.

### Which commands need the administrator capability

Reads stay open to a tenant, including the connector, model and purpose reads
the ordinary model picker uses. Writes to service configuration do not:

`SetApiKey`, `SetEmbeddingsSettings`, `SetPersistenceSettings`,
`SetDatabaseSettings`, `SetBackendTasksSettings`, `SetWsAuthSettings`,
`CreateConnection`, `UpdateConnection`, `DeleteConnection`,
`SetConnectionSecret`, `SetPurpose`, `AddMcpServer`, `RemoveMcpServer`,
`SetMcpServerEnabled`, `McpServerAction`, `UpsertMcpServer`, `SetMcpSecret`,
`UpsertServiceAccount`, `RemoveServiceAccount`.

`SetConfig` is mixed and is decided by its payload. A change that touches only
the seven personality traits is tenant work. A change that touches an
`embeddings_*` or `persistence_*` field needs the administrator capability.

Every other command is tenant work, including `ClearAllHistory`, which clears
only the calling user's conversations.

### Discovering your own capability

`GetConfig` reports it. `Config.caller_capability` is `"admin"` or `"tenant"`
for the connection that asked, and the `ConfigChanged` event carries the same
value. A client should read it and render the sections it may not change as
unavailable, with the reason, rather than letting a write fail on submit.

The field is absent from a daemon that predates the authorization tier. Absent
means "not reported", not "tenant": keep the prior behaviour in that case.

### What a refusal looks like on the wire

A refused command produces an ordinary error frame. The connection stays open
and keeps serving, and the command has no effect at all - the gate runs before
the handler.

```json
{
  "error": {
    "id": "req-7",
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

Branch on `detail.code`, never on the message text. `not_authorized` is stable.
The daemon also classifies `unsupported`, `not_found` and `already_terminal`;
an error it cannot classify honestly carries no `detail` at all, which a caller
must read as "unclassified" rather than "not an authorization problem".

`detail` is an **optional field on the existing `error` frame**, not a new frame
variant. An older client that knows only `{id, error}` parses it unchanged and
ignores the extra field, and the `error` string repeats `detail.description`. A
code an older client does not recognize round-trips rather than failing the
parse, so the daemon can add codes later.

### Behaviour change for existing API consumers

**This changes what an existing remote caller may do.** Before this tier, any
authenticated principal held the whole admin surface. Now a WebSocket caller is
a tenant unless `[authz] admin_subjects` names its subject, so integrations that
issue the commands listed above will start receiving `not_authorized`.

What an integrator must do:

- If your integration only sends messages and reads its own data, nothing
  changes.
- If it configures the daemon, add its authenticated subject (the JWT `sub`) to
  `[authz] admin_subjects` in `daemon.toml` and restart the daemon.
- Read `Config.caller_capability` at connect time and adapt, instead of
  discovering the boundary from a refusal.

Local clients on the same account are unaffected: the peer-uid grant makes them
administrators with no configuration.

## Implementation notes
- WebSocket is the remote-friendly transport; D-Bus remains best for local desktop integration.
- Both adapters should be covered by integration tests that replay the same command/event scenarios.
- WS auth (v1): Bearer JWT validated at handshake. Tokens are issued locally via D-Bus settings method.
- First-party clients should default to WS transport (localhost by default, configurable), while D-Bus remains available for host integration and bootstrap flows.

## Transport-level limits

Every transport into the daemon enforces a **4 MiB (`4 * 1024 * 1024 == 4_194_304` bytes)** ceiling on inbound payloads. The cap is identical across transports so a client that fits the smallest transport (UDS) trivially fits the others; it also keeps a single unauthenticated or compromised client from forcing a multi-tens-of-MB allocation per message.

| Transport          | Cap source                                                                                          | Behavior at over-cap                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| WebSocket          | `crates/ws-interface/src/lib.rs::MAX_WS_MESSAGE_BYTES` — applied as both `max_message_size` and `max_frame_size` on the `WebSocketUpgrade` | Server sends a close frame with RFC 6455 status code **1009** ("Message Too Big") and tears the connection down. |
| Unix domain socket | `crates/uds-interface/src/lib.rs::MAX_FRAME_LEN`                                                    | Length-prefix read returns `InvalidData`; the connection is closed.                                              |
| D-Bus bridge       | `crates/dbus-bridge/src/transport.rs::MAX_FRAME_LEN`                                                | Same as UDS — `InvalidData` and close.                                                                           |

If you raise the cap, raise it on **all three** in lockstep — otherwise a client that fits the largest transport will be silently truncated on the smallest, producing confusing partial-message errors. The cap is deliberately conservative; per-user / per-connection rate limiting is a separate concern tracked in `SECURITY_AUDIT.md` #5.
