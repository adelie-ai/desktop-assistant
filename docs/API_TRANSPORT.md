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

If settings later grow significantly, introduce `ListConfigSchema` and move to a registry-based key/value system.

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
