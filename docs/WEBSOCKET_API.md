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

For local clients, JWTs are typically minted via D-Bus:

- `org.desktopAssistant.Settings.GenerateWsJwt(subject)`
- Subject resolves to current OS username on the user bus.

For remote clients (no D-Bus), use `/login` with HTTP Basic auth to mint a bearer JWT:

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
- `set_embeddings_settings { connector?, model?, base_url? }`
- `get_connector_defaults { connector }`
- `get_persistence_settings`
- `set_persistence_settings { enabled, remote_url?, remote_name?, push_on_update }`

Result payloads are typed variants (`pong`, `status`, `conversation_id`, `conversations`, `conversation`, `config`, `ack`, etc.).

## Events

Current event variants:

- `config_changed { config }`
- `assistant_delta { conversation_id, request_id, chunk }`
- `assistant_completed { conversation_id, request_id, full_response }`
- `assistant_error { conversation_id, request_id, error }`

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

## Notes

- The command `id` correlates only `result`/`error` frames.
- Streaming assistant events are correlated by server-generated `request_id` inside event payloads.
- Multiple requests can be in flight concurrently; clients should match by `id` and event metadata.
