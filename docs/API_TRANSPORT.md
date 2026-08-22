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

`SetApiKey`, `SetEmbeddingsSettings`, `SetDatabaseSettings`,
`SetBackendTasksSettings`, `SetWsAuthSettings`,
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

`ListSkills` and `SetSkillApproval` are tenant work. The listing is the
caller's own view of the catalog - the host-global skills plus their own - and
the write reaches only rows this person owns, so neither decides anything for
another tenant or for the host. Approving is a person's act rather than the
model's, on the same footing as clearing a negative memory: it is a command,
which arrives from an authenticated client connection and never from a turn,
and no tool the model is offered reaches it. An assistant able to approve its
own procedures would be recording the user's consent on the user's behalf.

`ListContextBreakdowns` and `GetContextBreakdown` are tenant work. Each reads
what filled the caller's own turns, scoped by user id in storage the same way
their conversations are, and a turn's correlation id grants nothing: another
user's turn reads as absent rather than as a refusal. `docs/WEBSOCKET_API.md`
documents the payload, including the one rule an integrator must not break -
the assembler's per-part estimate and the provider's own reported count are two
measurements of one prompt and are never summed or derived from each other.
Both commands reach every transport, D-Bus included, because
`org.desktopAssistant.Commands.SendCommand` forwards any command it does not
explicitly refuse.

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

## The UDS handshake: declaring what you share about the user

The first frame on a Unix-socket connection is a JSON handshake. Every field in
it is optional, so the historic `{"jwt": "..."}` still parses.

| Field | Purpose |
|---|---|
| `jwt` | Bearer token. A local client authenticates by kernel peer credentials and sends none. |
| `system_id` | The client's per-machine id, for exact tool co-location. |
| `host_label` | A friendly name for that machine, for the remote-tool note. |
| `client_context` | What the client knows about the user and the device: name, username, home directory, hostname, timezone, OS. It grounds the system prompt. |
| `share_client_context` | The client's declaration about the field above. |

A client that reports no `client_context` gets one anyway: the daemon reads the
kernel-attested identity of the connecting process and grounds the prompt with
that user's name, login and home directory. This is correct for a desktop client
that runs as the person using it, and wrong for a client that connects for
somebody else - a server-side client serving remote users describes its own host,
not the person it serves.

`share_client_context: false` is how a client refuses that substitution. The
daemon then attaches no client context to the connection at all, and the system
prompt carries no block about the user or the device. The kernel peer identity
still authenticates the connection; it only stops grounding the prompt.

An absent `share_client_context`, and an explicit `true`, both mean "no
refusal", and behave exactly as the daemon behaved before the field existed. A
client that shares its context sends no such field, so its handshake bytes are
unchanged.

### What an integrator must do

- A client that runs as the person it serves needs no change.
- A client that connects on behalf of other people sends
  `share_client_context: false`, and supplies each real user's context per turn
  on `SendMessage.client_context` instead. Without the declaration the daemon
  substitutes the connecting process's own identity whenever a turn carries no
  per-turn context.
- The WebSocket door has no such substitution to refuse: it reads the client
  context from an upgrade header and infers nothing when the header is absent.

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

### Outbound: the response byte budget

The same cap applies to what the daemon writes back, and the daemon holds itself to it in two places.

**A response is bounded where it is built.** `get_conversation` and `get_messages` cut their message list to `crates/api-model/src/lib.rs::MAX_RESPONSE_BYTES` (3 MiB) of **serialized** bytes, measured as the encoder emits them - JSON escaping inflates a control byte six times, so a raw content length is not a bound. The budget sits a megabyte under the transport cap because the layer that builds a response cannot see the frame a transport wraps it in. `uds-interface` and `ws-interface` each hold the two constants together with a compile-time assertion.

**The whole `list_conversations` answer is bounded, not one field of it.** A conversation title is bounded (below), and so are a summary's `tags`, which the client writes and nothing validates - three conversations carrying a 2 MiB tag each produced an answer of 6,291,823 bytes, past the 3,145,728-byte budget and past the 4,194,304-byte transport cap, so the conversation list could not be read at all. The answer is cut as a whole to `MAX_RESPONSE_BYTES`, keeping the FRONT of the list, which is the most recently updated part and what a sidebar shows first. A row that does not fit loses tags from the end of its tag list before it loses its place in the list, because a conversation the caller cannot see at all is worse than one shown with fewer tags.

**A title is bounded where it is WRITTEN.** A conversation title rides in every `get_conversation` answer, every `list_conversations` row, and the `conversation_title_changed` event, so one large title could take the whole response budget on its own: every message was dropped and the answer was oversize anyway, so the conversation could not be opened and the conversation list could not be read. The cap is `desktop_assistant_core::domain::MAX_TITLE_BYTES` (4 KiB) of **serialized** bytes, and it is applied two different ways, because the two sources of a title need different answers:

- **A title a client supplies is refused past the cap.** `create_conversation` and `rename_conversation` return a classified decline carrying the business code `conversation_title_too_long`, a description that names the size only, a message fit to show the person who typed the title, and `retryable: false` - the same input is always refused. Nothing is stored and nothing already stored is changed. It is a refusal rather than a truncation because silently rewriting what a person typed is the loss this bound exists to remove, and a title cut on the way in cannot be recovered while a title refused on the way in costs one retry.
- **A title the daemon composes for itself is cut at generation.** `Standalone: <name>`, `Subagent: <name>`, and the name an LLM writes after the first message of a conversation. Refusing there would fail an operation the user did not ask for. A cut title ends with `...`, and where the title is composed the label leads, so what a cut removes is the supplied name rather than the label.

4 KiB is chosen for the person who types a title, not for the response envelope: it holds about 4090 Latin characters, about 1360 characters of a three-byte script, or about 680 characters if every one of them escapes to six bytes - far more than a one-line label needs in any script.

**The response still cuts an over-cap title, as a backstop for rows written before that rule.** A write rule cannot repair a title that is already stored, so `get_conversation` and `list_conversations` cut such a title to `crates/api-model/src/lib.rs::MAX_TITLE_BYTES`, state the loss in words at the end of what is kept, and report the stored size in `title_total_bytes`. The two constants are the same number, held equal by a compile-time assertion, so a title accepted on the way in is never cut on the way out. A title inside the bound is returned untouched.

A cut response is a **normal, successful result that says it is partial**, never a transport failure:

| Field | Where | Meaning |
| --- | --- | --- |
| `omitted_leading_messages` | `ConversationView` | How many of the OLDEST messages were left out. `0` (omitted from the wire) means no message was left out - **not** that the response is whole, because a single over-budget message comes back headed with the count still at zero. It is also a cursor: the omitted messages are raw indexes `0 .. omitted_leading_messages`, so read them with `get_messages { after_count: 0 }` and walk forward with `next_after_count`. |
| `size_capped` | `MessagesView` | The byte budget removed messages from this window. In tail mode `truncated` is set too, because tail mode drops the oldest. In cursor mode (`after_count >= 0`) only `size_capped` is set, because a cursor window drops the NEWEST end so the caller continues forward. `false` means no message was removed - **not** that the window is whole, because a window whose only message is past the whole budget comes back headed with the flag still `false`, so check `content_total_bytes` as well. In cursor mode it is the stop condition of a walk: `false` means there is no next page. |
| `next_after_count` | `MessagesView` | The raw message index this window ended at. In **cursor mode** (`after_count >= 0`), send it as the next request's `after_count`. Do **not** derive the cursor from `after_count + messages.len()`: `after_count` counts raw rows while `messages` counts what the `include_roles` filter let through, so a derived cursor names an earlier row than the window reached and re-reads rows under a role filter. `total_raw_count` when the window is empty. Each cut page carries at least one message, so the cursor always moves forward and a walk ends. In **tail mode** it names the end of the conversation, because a tail window keeps the newest messages - so following it returns an empty window and reads none of what the tail dropped. A tail caller reads `truncated` instead, then pages forward in cursor mode from `after_count: 0`. |
| `title_total_bytes` | `ConversationView`, `ConversationSummary` | The whole stored title is this many bytes and you were given only the beginning of it. Absent means the title is whole. Only a title stored before the write bound can carry it, because a title written from now on is inside the cap. **A client that pre-fills a rename field from a served title should still check it**: the value carries a notice on the end, so writing it back with `rename_conversation` would offer the cut value - and that rename is now refused rather than stored, so the check turns a refusal the person cannot act on into a field the client can disable. The notice is prose, and prose cannot be told from a title the user typed - this field can. |
| `omitted_trailing_conversations` | `ConversationSummary` | How many conversations the `list_conversations` answer left out, from the END of the list. `0` (omitted from the wire) means the list is whole. Carried on every returned row, with the same value on each, because the result is a bare list with no envelope to put it in - read it off any row. |
| `omitted_tags` | `ConversationSummary` | How many of THIS row's tags were left out to keep the row inside the budget. `0` (omitted from the wire) means the row carries every tag it has. `tags` keeps the first ones. |
| `content_total_bytes` | `MessageView` | The whole stored content is this many bytes and you were given only the beginning of it. Absent means the content is whole. Set only when one message alone is past the budget: the row is headed rather than dropped, so a conversation is never unreadable because of one large message. The content states the same loss in words at its front. Nothing was deleted - the message is stored whole, and no byte-range read exists yet. |

All of them are additive fields with serde defaults, so an older payload parses unchanged and an older client ignores them. Every partial marker is omitted at its default, so a whole response keeps the wire bytes it always had; `next_after_count` is always present, because a cursor that disappears at zero is a trap rather than a saving.

The `conversation_title_changed` event carries a bounded title and has no field for the marker, so a client cannot tell a cut title from a whole one there. A client that pre-fills a rename field takes the title from `get_conversation` or `list_conversations`, never from that event.

**A client holds itself to the same cap on the way out.** `UdsClient::send_command` refuses a request larger than `MAX_FRAME_LEN` and `WsClient::send_command` refuses one larger than `MAX_WS_MESSAGE_BYTES`, each before the request is enqueued. The peer answers an oversize message by closing the connection, which fails every request in flight, so one unsendable request would otherwise cost the caller its whole connection. It now costs the caller that one call, with an error naming the size.

**A frame past the cap is never written.** `desktop_assistant_frame_codec::write_frame` refuses a body larger than `read_frame` would accept, before writing any bytes, and each transport turns that refusal into a failure of **the one request**, not of the connection: the caller gets an `Error` frame carrying its own request id, and the connection keeps serving. An `Event` carries no request id, so an oversize one is dropped with a warning rather than taken as a reason to disconnect. Dropping a turn's terminal event leaves a client waiting for a signal that never comes; a substitute event is tracked separately. Where the request id is itself so large that even the error frame would not fit, that frame is dropped too and the one request goes unanswered - the caller's own request timeout covers it, and the connection survives.
