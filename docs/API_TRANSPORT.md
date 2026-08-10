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
- `SendMessage { conversation_id?, content, ..., turn_id?, traceparent? }` (streaming response)
  - `turn_id` is the client's own correlation id for this turn, as a uuid. The
    daemon adopts a well-formed, non-nil value and returns it on the ack;
    anything else, including an absent field, means the daemon mints its own.
    It grants nothing and is not the idempotency key.
  - `traceparent` is a W3C trace context for a caller that is already inside a
    trace, so the daemon continues that trace rather than starting one. Absent
    or unusable values are discarded and never fail the turn.
- `GetConfig`
- `SetConfig { changes }`

### Events
- `StatusChanged(Status)`
- `ConfigChanged { config }`
- `MessageStarted { message_id, role }` (optional)
- `AssistantDelta { request_id, chunk }`
- `AssistantCompleted { request_id, full_response }`
- `Error { code, message, retryable }`
- `AssistantStatus { conversation_id, request_id, message, capability_change? }`

`capability_change` carries a tool-provenance narrowing: once a turn has taken
in content from outside the trust boundary, the daemon refuses the acting tool
tiers for the remainder of that turn, and says so once. It is optional and
absent on ordinary progress statuses. `docs/WEBSOCKET_API.md` documents the
payload and what an integrator does about it; the same contract reaches D-Bus
clients through `ResponseStatus`, minus the structured field
(`docs/dbus-api.md`).

A conversation can turn that gate off for itself with
`SetConversationToolGate { conversation_id, disabled }`. With the override
on, a gated tool runs even after the turn ingests outside content, and the
one-time status line reads differently - "Live dangerously is on for this
conversation..." rather than the closed-gate line - and carries no
`capability_change`, because nothing actually closed. `docs/WEBSOCKET_API.md`
documents the full wire shape and the fail-closed resolution rule.

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

Because the subject now decides an authorization outcome, a connection that
cannot be given one is refused rather than given a fallback. A bearer token that
validates but carries no `sub` claim (or a blank one) is rejected on both the
WebSocket door and the UDS token fallback, instead of resolving to the storage
sentinel `"default"`. That sentinel is also not admissible to
`[authz] admin_subjects`: it names a schema default, not a person, and putting
it on the list would have promoted every subject-less connection.

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
   administrator. **No command writes the section** - it is set by editing the
   file. Editing it needs a daemon restart, and a reload reports `authz` in
   `restart_required`.

   The API surface is therefore closed, but that is not the same as the file
   being unreachable. The daemon-side `fileio`, `terminal` and `command` MCP
   servers run in the daemon process as the daemon's own uid, so a tool call in
   an ordinary turn could write `daemon.toml` and wait for a restart. Those
   three ship disabled, and constraining daemon-side tool execution is the
   tool-execution half of `docs/design/multi-tenancy-boundary.md` (decisions 3
   to 5), not part of this tier. Enable them on a shared instance only with
   that in mind.

### Which commands need the administrator capability

Reads stay open to a tenant, including the connector, model and purpose reads
the ordinary model picker uses. Writes to service configuration do not:

`SetApiKey`, `SetEmbeddingsSettings`, `SetPersistenceSettings`,
`SetDatabaseSettings`, `SetBackendTasksSettings`, `SetWsAuthSettings`,
`CreateConnection`, `UpdateConnection`, `DeleteConnection`,
`SetConnectionSecret`, `SetPurpose`, `AddMcpServer`, `RemoveMcpServer`,
`SetMcpServerEnabled`, `UpsertMcpServer`, `SetMcpSecret`,
`UpsertServiceAccount`, `RemoveServiceAccount`, `StartKnowledgeMaintenance`,
and `McpServerAction` for every verb except `"status"`.

`SetConfig` is decided by its payload, and every field it carries today writes
daemon-global state, so any non-empty change set needs the administrator
capability. That includes the seven personality traits: they are one global
`[personality]` block in `daemon.toml`, and every conversation without an
override resolves against it, so one caller writing a trait changes every other
tenant's assistant. A tenant sets their own disposition with
`SetConversationPersonality`, which is genuinely per-conversation and stays
tenant work. This is staging, not a judgement that personality is an operator
concern: the per-user override layer (#986) is where the traits belong, and when
it lands a tenant edits their own while the instance default stays admin.

`SetConversationToolGate` is a per-conversation, tenant-level lever from the
start: unlike the personality traits it has no global counterpart in
`daemon.toml` to weigh against, so it stays tenant work with no staging
period.

`McpServerAction` is split the same way, by its verb. `"status"` is a read
returning exactly what `ListMcpServers` returns and stays tenant; `"start"`,
`"stop"`, `"restart"` and any verb added later spawn or kill a child process and
need the administrator capability.

`StartKnowledgeMaintenance` needs the administrator capability. It looks like
knowledge work, which is otherwise per-user, but every operation reaches past
the caller: the embedding recompute nulls and re-derives vectors for **all**
tenants through the operator's provider, consolidation rewrites every user's
knowledge base, and extraction's archival phase widens to all users under the
default-subject sentinel.

`ListNegativeMemories`, `GetNegativeMemory` and `ClearNegativeMemory` are
tenant work. What one person's assistant tried and how it failed is that
person's, on the same footing as their knowledge and their scratchpads, and
requiring the administrator capability would put a single-user desktop's own
memory out of its owner's reach on any multi-tenant deployment. Clearing is
still a person's judgement rather than the model's: it is a command, which
arrives from an authenticated client connection and never from a turn, and no
tool the model is offered reaches it.

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
  `[authz] admin_subjects` in `daemon.toml` and restart the daemon. `"default"`
  is not a valid entry - it is the storage sentinel, and the daemon drops it with
  a warning.
- Make sure the token carries a `sub`. An issuer that mints a token without one
  now fails the handshake with `401` instead of connecting as `"default"`.
- Read `Config.caller_capability` at connect time and adapt, instead of
  discovering the boundary from a refusal.
- Handle `429` from `POST /login`: the endpoint throttles attempts per source
  address and per username, and names the wait in `Retry-After`. Honour that
  wait rather than retrying on your own schedule - an early retry spends from
  the same budget and pushes the wait out again. It is not a credential verdict,
  so do not re-prompt for the password on it.

Local clients on the same account are unaffected: the peer-uid grant makes them
administrators with no configuration.

### `evicted_results` on `ToolUsageView` (behaviour change)

**This changes what an existing reader of the tool-usage aggregate sees.**

A completed agentic step drops its large tool results from the model's view
and reads them as a short pointer to the scratchpad note it distilled them
into. That decision is now recorded on the message row, so every later turn
reads the pointer while the stored transcript keeps every byte the tool
returned. Those rows are counted in `evicted_results`.

Before, `evicted_results` could only be non-zero for a conversation compacted
by an old build that overwrote the row. A non-zero count therefore meant "the
bytes are gone and `result_bytes` under-reports what this tool cost". That is
no longer what it means on its own.

What an integrator must do:

- Read `evicted_results` as "results the model reads as a pointer", not as
  "results whose bytes are lost".
- Do not add `evicted_results` to `result_bytes` to reconstruct an original
  size. For a row evicted by a completed step the bytes are already counted in
  full, because the conversation still holds them; for a row from an old build
  they are unrecoverable and count zero. The two cannot be told apart in this
  view.
- A rising `evicted_results` with steady `result_bytes` is the normal, healthy
  shape for a conversation doing long agentic work. It is not data loss.

No field was added or removed, and the wire bytes are unchanged.

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
