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

### Settings Methods

- `GetLlmSettings() -> (connector: s, model: s, base_url: s, has_api_key: b)`
- `SetLlmSettings(connector: s, model: s, base_url: s) -> ()`
- `SetApiKey(api_key: s) -> ()`
- `GenerateWsJwt(subject: s) -> token: s`
  - Token subject is always the current OS username on the user bus.
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
`SetLlmSettings`, `SetEmbeddingsSettings`, `SetDatabaseSettings`,
`SetBackendTasksSettings`, `SetWsAuthSettings`, the connection and purpose
writes, the MCP lifecycle writes, and `SetConfig` (whose personality traits are
one global block, not a per-user preference). Reads and conversations are
unaffected, and a tenant still sets their own disposition per conversation.

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
  - `GetLlmSettings` and `GetEmbeddingsSettings` only return non-sensitive fields plus `has_api_key`.
- WebSocket auth uses bearer JWTs:
  - Generate locally signed tokens with `GenerateWsJwt`.
  - Multiple tokens can be valid at once until expiry.
