# D-Bus API

> Deployment + the `adelie-dbus-bridge` that serves this surface during the
> cutover: see [dbus-bridge.md](dbus-bridge.md).

Service name: `org.desktopAssistant`  
Object path: `/org/desktopAssistant/Conversations`  
Interface: `org.desktopAssistant.Conversations`

Settings object path: `/org/desktopAssistant/Settings`  
Interface: `org.desktopAssistant.Settings`

## Methods

- `CreateConversation(title: s) -> id: s`
- `ListConversations(max_age_days: i) -> a(ssus)`
  - `max_age_days = 0` disables filtering
  - tuple: `(id, title, message_count, updated_at)` where `updated_at` is `YYYY-MM-DD HH:MM:SS`
- `GetConversation(id: s) -> (id: s, title: s, messages: a(ss))`
  - message tuple: `(role, content)`
- `DeleteConversation(id: s) -> ()`
- `ClearAllHistory() -> deleted_count: u`
- `SendPrompt(conversation_id: s, prompt: s) -> request_id: s`
  - The returned `request_id` is what every streamed event for this turn
    carries, and it is also the turn's trace id in the daemon's telemetry.
  - A D-Bus caller cannot supply its own id: the signature has no room for one
    and no options dictionary to add one to, so the bridge mints. The bridge is
    therefore the top of the trace for a D-Bus caller, and the caller still
    correlates by the value it is handed back.
- `SendPromptWithSystemRefinement(conversation_id: s, prompt: s, system_refinement: s) -> request_id: s`
  - As `SendPrompt`, plus a system-prompt addition that applies to this turn
    only. It is never stored and never appears in chat history.
- `SetConversationToolGate(conversation_id: s, disabled: b) -> b`
  - Per-conversation override for the tool-provenance gate (see "Tool-provenance
    gating" below). Returns the stored value after the write.

### Settings Methods

- `SetApiKey(api_key: s) -> ()`
- `GetEmbeddingsSettings() -> (connector: s, model: s, base_url: s, has_api_key: b, available: b, is_default: b)`
- `SetEmbeddingsSettings(connector: s, model: s, base_url: s) -> ()`
  - Empty `connector` clears the override and reverts to defaulting from the LLM connector
- `GetConnectorDefaults(connector: s) -> (llm_model: s, llm_base_url: s, embeddings_model: s, embeddings_base_url: s, embeddings_available: b)`
  - Returns provider defaults for the requested connector (empty `connector` resolves to the default connector)
- `GetConfig() -> (llm_connector: s, llm_model: s, llm_base_url: s, llm_has_api_key: b, embeddings_connector: s, embeddings_model: s, embeddings_base_url: s, embeddings_has_api_key: b, embeddings_available: b, embeddings_is_default: b, persistence_enabled: b, persistence_remote_url: s, persistence_remote_name: s, persistence_push_on_update: b)`
  - `persistence_remote_url` is redacted: an inline password reads `***` (see
    [Credentials in connection URLs](#credentials-in-connection-urls)).
- `SetConfig(changes: ConfigPatchArgs) -> same tuple as GetConfig`
  - `ConfigPatchArgs` is a struct of `(set_*, value)` pairs so callers can change only selected fields.
  - String values are only applied when their corresponding `set_*` flag is `true`.
  - For optional string fields, passing an empty string with `set_* = true` clears the field where supported.

### Credentials in connection URLs

Every connection URL the daemon returns — the database `url`, the git
`persistence_remote_url` — has its password replaced by `***`. The rest of the
URL (scheme, user, host, port, database, options) is intact, so a settings UI
can still show what is configured.

Writes round-trip: posting back a URL that still carries the `***` placeholder
keeps the stored password, provided nothing else in the URL changed. Editing
any other part of the URL means re-entering the password — the daemon refuses
to splice its stored credential into a URL a client has edited, and never
stores the placeholder as if it were a password.

## Authorization

The D-Bus surface is served by `adelie-dbus-bridge`, which reaches the daemon
over the local Unix socket. The daemon authorizes that connection by the
bridge's kernel-attested peer uid: a bridge running as the daemon's own account
holds the **administrator** capability, so on a single-user desktop every method
here works exactly as it did, with no configuration to add.

Where the bridge runs as a different account than the daemon, it is a **tenant**
unless `[authz] admin_subjects` in `daemon.toml` names that account's login
name. A tenant is refused the service-configuration writes - `SetApiKey`,
`SetEmbeddingsSettings`, `SetDatabaseSettings`, `SetBackendTasksSettings`,
`SetWsAuthSettings`, the connection and purpose
writes, the MCP lifecycle writes, and `SetConfig` (whose personality traits are
one global block, not a per-user preference). Reads and conversations are
unaffected, and a tenant still sets their own disposition per conversation,
and their own tool-provenance-gate override (`SetConversationToolGate`).

A refusal reaches D-Bus as an `org.freedesktop.DBus.Error.Failed` whose message
begins `not authorized:` and names the command. The structured classification
the socket transports carry (`detail.code = "not_authorized"`) does not survive
the D-Bus error type, which carries only a name and a message (#974). See
[API_TRANSPORT.md](API_TRANSPORT.md) for the full contract.

## Signals

- `ResponseChunk(conversation_id: s, request_id: s, chunk: s)`
- `ResponseComplete(conversation_id: s, request_id: s, full_response: s)`
- `ResponseError(conversation_id: s, request_id: s, error: s)`
- `ConfigChanged(llm_connector: s, llm_model: s, llm_base_url: s, llm_has_api_key: b, embeddings_connector: s, embeddings_model: s, embeddings_base_url: s, embeddings_has_api_key: b, embeddings_available: b, embeddings_is_default: b, persistence_enabled: b, persistence_remote_url: s, persistence_remote_name: s, persistence_push_on_update: b)`

### Tool-provenance gating (behaviour change)

A turn that has read content from outside the trust boundary refuses the tools
that could act on it - write, network egress, code execution, and anything the
daemon cannot classify - for the remainder of that turn. A tool call that would
previously have run may now come back refused. The refusal is an ordinary tool
result, not an error, so the turn continues and the assistant takes another
path; the next turn starts clean.

The daemon emits one status for the turn when this happens, carrying a
human-readable line. D-Bus clients see it on the `Status(conversation_id: s,
request_id: s, message: s)` signal, like any other progress message. The machine-readable form of the same fact -
`capability_change`, naming the cause and the closed tiers - rides the
WebSocket `assistant_status` event and is **not** projected onto the D-Bus
signal, which carries only the text. A D-Bus client that needs the structured
value reads it over the WebSocket transport instead
(`docs/WEBSOCKET_API.md`).

A conversation can turn the gate off for itself: `SetConversationToolGate`
above. With the override on, a gated tool runs even after the turn reads
outside content, and the `Status` signal instead carries "Live dangerously is
on for this conversation: a tool that would normally be refused after reading
outside content ran anyway." - once per turn, the same channel as the
closed-gate line. Fails closed everywhere a lookup can fail: an unset value,
a missing conversation row, a cross-user row, or a store error all resolve
to the gate staying enforced.

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

## Quick `busctl` examples

```bash
# list conversations
busctl --user call org.desktopAssistant \
  /org/desktopAssistant/Conversations \
  org.desktopAssistant.Conversations ListConversations i 7

# create conversation
busctl --user call org.desktopAssistant \
  /org/desktopAssistant/Conversations \
  org.desktopAssistant.Conversations CreateConversation s "My Chat"
```

## Behavior Notes

- `SendPrompt` is asynchronous: method returns request id immediately.
- Streaming lifecycle is signal-driven (`Chunk*`, then `Complete` or `Error`).
- IDs are UUID-style strings generated by daemon.
- Secret handling over D-Bus is write-only by design:
  - API keys can be written with `SetApiKey`.
  - There is no method that returns secret values.
  - `GetEmbeddingsSettings` only returns non-sensitive fields plus `has_api_key`.
- WebSocket auth uses bearer JWTs, minted off the D-Bus surface entirely; see
  [WEBSOCKET_API.md](WEBSOCKET_API.md). Multiple tokens can be valid at once
  until expiry.
- The skill catalog is **not on this surface**. `list_skills` and
  `set_skill_approval` are what let a person see the skills the assistant wrote
  for itself and approve one for use (see
  [WEBSOCKET_API.md](WEBSOCKET_API.md)). This bridge carries a curated subset
  of the command surface and has no typed method for them, so a D-Bus client
  reaches them through the generic `org.desktopAssistant.Commands.SendCommand`
  channel or not at all. Nothing an existing D-Bus caller does changes; what
  such a client cannot do without that channel is show a person a skill waiting
  for approval.
- Negative memory is **not on this surface**. The daemon holds a tool call back
  when the same act went badly before, and `list_negative_memories`,
  `get_negative_memory` and `clear_negative_memory` are what let a person see
  and overrule that (see [WEBSOCKET_API.md](WEBSOCKET_API.md)). This bridge
  carries a curated subset of the command surface and has no method for them
  yet, so a D-Bus client cannot list or clear one. Nothing an existing D-Bus
  caller does changes; what such a client cannot do is show a person why a call
  was held.
