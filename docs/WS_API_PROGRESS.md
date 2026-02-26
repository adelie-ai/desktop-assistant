# WebSocket API work — TODO & Progress

Branch: `feature/ws-api`

## Goal
Expose the same small API surface over **WebSocket** as currently exists/exists conceptually over **D-Bus**, using common handlers + thin adapters.

## Current status
- [x] Doc: `docs/API_TRANSPORT.md` (shared surface + approach)

## TODO (planned pieces)

### 0) Repo hygiene / discovery
- [x] Identify existing D-Bus interface methods/signals relevant to: status, send message, config
- [x] Decide whether to reuse existing types or introduce new shared API model crate (added `crates/api-model`)

### 1) Shared API model (canonical types)
- [x] Add crate `crates/api-model` with serde types:
  - `Command`
  - `CommandResult`
  - `Event`
  - view structs for conversations + settings
- [~] Unit tests for (de)serialization stability
  - [x] basic JSON roundtrip tests
  - [ ] add golden JSON fixtures to lock schema (recommended)

### 2) Application handlers (common logic)
- [x] Add crate `crates/application` with:
  - `trait AssistantApiHandler` (async)
  - `DefaultAssistantApiHandler` delegating to existing inbound ports
  - `EventSink` trait for adapters to forward canonical events
- [x] Unit tests: ping + send message streaming emits deltas then completion

### 3) WebSocket adapter
- [x] Add crate `crates/ws-interface`
- [x] Implement WS server (axum):
  - accept connections on `/ws`
  - request/reply for non-streaming commands
  - `SendMessage` spawns streaming handler and forwards canonical events
- [x] Integration tests for WS protocol (connect + ping)
- [x] Integration tests for WS protocol (get status)
- [x] Integration tests for WS protocol (`SendMessage` ACK + streaming events)

### 4) Wire into daemon
- [x] Add env/config for WS bind address + port (default localhost): `DESKTOP_ASSISTANT_WS_BIND` (e.g. `127.0.0.1:11339`)
- [x] Start WS server in daemon main (spawned task)
- [x] Ensure clean shutdown (SIGINT/SIGTERM -> WS graceful shutdown)

### 5) Config endpoints parity
- [x] Define typed config struct (~10 settings)
- [x] Implement `GetConfig`/`SetConfig`
- [x] Emit `ConfigChanged`

### 6) Chat streaming parity
- [x] Implement `SendMessage` streaming (`AssistantDelta`, `AssistantCompleted`)
- [x] Backpressure + cancellation

### 7) WebSocket auth
- [x] Require bearer token at WS handshake (`Authorization: Bearer <jwt>`)
- [x] Add D-Bus JWT issuance method (`GenerateWsJwt`)
- [x] Validate JWT in daemon before upgrading socket
- [ ] Trusted external issuer support via config (follow-up)

### 8) Client transport defaulting
- [x] TUI defaults to WebSocket transport (`DESKTOP_ASSISTANT_TUI_TRANSPORT=ws`)
- [x] TUI supports explicit D-Bus transport override (`DESKTOP_ASSISTANT_TUI_TRANSPORT=dbus`)
- [x] TUI bootstraps WS JWT via D-Bus when token env is not provided
- [x] Move KDE widget transport to same defaulting model (ws/dbus, ws URL + subject config)

## Progress log

### 2026-02-25
- Surveyed existing D-Bus APIs:
  - `org.desktopAssistant.Conversations`: create/list/get/delete/clear, plus `send_prompt` streaming via signals
  - `org.desktopAssistant.Settings`: LLM + embeddings + persistence settings, write-only API key
- Added `crates/api-model` with canonical `Command` / `CommandResult` / `Event` types and view structs.
  - Naming: `SendMessage` (not `SendPrompt`), streaming events `AssistantDelta`/`AssistantCompleted`/`AssistantError`.
- Added `crates/application` with protocol-neutral handlers that map `Command` to existing core inbound ports, plus a streaming `handle_send_message` that emits canonical `Assistant*` events via an `EventSink`.

- Created branch `feature/ws-api`.
- Added docs: `docs/API_TRANSPORT.md`.
- Created this progress tracker.

### 2026-02-26
- Added WS integration tests for:
  - `GetStatus` roundtrip
  - `SendMessage` ACK + `AssistantDelta` + `AssistantCompleted` event sequence
- Added `serve_with_shutdown(...)` in `crates/ws-interface` using Axum graceful shutdown.
- Updated daemon main loop to:
  - wait for `SIGINT` / `SIGTERM`
  - trigger WS shutdown via oneshot signal
  - await WS task completion before exiting
- Added bounded event buffering and disconnect-aware cancellation in WS streaming:
  - bounded outbound WS frame queue in adapter
  - bounded chunk queue in application handler
  - stream callback aborts when queue is full or client sink disconnects
- Added tests for cancellation:
  - application unit test (`send_message_cancels_when_sink_disconnects`)
  - WS integration test (`ws_send_message_cancels_when_client_disconnects`)
- Added typed transport config model:
  - `Command::{GetConfig, SetConfig}`
  - `CommandResult::Config`
  - `Event::ConfigChanged`
- Implemented config parity in application handlers:
  - aggregate `GetConfig` over existing settings ports
  - apply `SetConfig` partial updates over existing settings ports
- Added config parity tests:
  - application tests for `GetConfig` and `SetConfig`
  - WS integration test for `SetConfig` result + `ConfigChanged` event
- Added D-Bus config parity follow-up on `org.desktopAssistant.Settings`:
  - `GetConfig` aggregate method
  - patch-based `SetConfig(changes)` method
  - `ConfigChanged` signal emission after successful update
- Pivoted WS auth from opaque API keys to local JWTs:
  - Added `GenerateWsJwt(subject)` on D-Bus settings interface.
  - Added daemon local JWT issuer/signing-key management.
  - Enforced `Authorization: Bearer <jwt>` on `/ws` handshake.
  - Added WS tests for missing/invalid token rejection.
- Updated TUI to prefer WS by default:
  - Added WS transport client using canonical WS protocol (`Command`/`WsFrame`).
  - Added transport config via env (`DESKTOP_ASSISTANT_TUI_TRANSPORT`, `DESKTOP_ASSISTANT_TUI_WS_URL`, `DESKTOP_ASSISTANT_TUI_WS_JWT`, `DESKTOP_ASSISTANT_TUI_WS_SUBJECT`).
  - Added equivalent TUI CLI flags via `clap` (`--transport`, `--ws-url`, `--ws-jwt`, `--ws-subject`) with args taking precedence.
  - Added D-Bus settings JWT bootstrap when WS JWT is not provided.
- Updated KDE shared widget transport helper:
  - Added named connection profile support (`local` D-Bus + named WS remotes) with global default selection.
  - Added helper CLI/env support for `--connection-name` / `DESKTOP_ASSISTANT_WIDGET_CONNECTION`.
  - Kept legacy transport overrides (`transport`, `ws_url`, `ws_subject`) for backward compatibility.
  - Added local JWT bootstrap via D-Bus `GenerateWsJwt` when WS token is not explicitly configured.
  - Updated widget UIs to select a named connection in per-widget config.
  - Added KCM Connections tab to manage named profiles + global default.
- Added remote auth bootstrap endpoint:
  - Added HTTP `POST /login` endpoint on the WS server.
  - Uses Basic auth (`username:password`) and returns a bearer JWT with `sub=username`.
  - Local Linux default authenticates against OS account password; container/remote uses daemon env credentials (`DESKTOP_ASSISTANT_WS_LOGIN_USERNAME`, `DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD`).
  - D-Bus `GenerateWsJwt` now issues tokens with the current OS username as subject.
