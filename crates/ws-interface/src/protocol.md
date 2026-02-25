# WebSocket protocol (v1)

Endpoint: `ws://<host>:<port>/ws`

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
