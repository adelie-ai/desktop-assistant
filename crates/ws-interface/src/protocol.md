# WebSocket protocol (v1)

Endpoint: `ws://<host>:<port>/ws`

Handshake auth:
- Require `Authorization: Bearer <jwt>` header.
- JWT must be issued by the daemon's local issuer (v1).

Client -> server:

```json
{ "id": "<client-generated>", "command": { "ping": {} } }
```

Server -> client frames:

- Result:
```json
{ "result": { "id": "...", "result": { "pong": { "value": "pong" } } } }
```

- Error:
```json
{ "error": { "id": "...", "error": "..." } }
```

- Event (async):
```json
{ "event": { "event": { "assistant_delta": { "conversation_id": "...", "request_id": "...", "chunk": "..." } } } }
```

Notes:
- `SendMessage` returns an immediate `Ack` result and then emits `event` frames for streaming.
- `SetConfig` returns a `Config` result and then emits `ConfigChanged { config }`.
