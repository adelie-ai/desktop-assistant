# WebSocket API work — TODO & Progress

Branch: `feature/ws-api`

## Goal
Expose the same small API surface over **WebSocket** as currently exists/exists conceptually over **D-Bus**, using common handlers + thin adapters.

## Current status
- [x] Doc: `docs/API_TRANSPORT.md` (shared surface + approach)

## TODO (planned pieces)

### 0) Repo hygiene / discovery
- [ ] Identify existing D-Bus interface methods/signals relevant to: status, send message, config
- [ ] Decide whether to reuse existing types or introduce new shared API model crate

### 1) Shared API model (canonical types)
- [ ] Add crate `crates/api-model` (or similar) with serde types:
  - `Command`
  - `CommandResult`
  - `Event`
  - IDs/correlation (`request_id`, `message_id`, etc.)
- [ ] Unit tests for (de)serialization stability (golden JSON examples)

### 2) Application handlers (common logic)
- [ ] Add crate `crates/application` (or module) with:
  - `trait AssistantApiHandler` (async)
  - Implementations that delegate to existing domain/services
- [ ] Unit tests: given a command, emits expected results/events

### 3) WebSocket adapter
- [ ] Add crate `crates/ws-interface` (or similar)
- [ ] Implement WS server:
  - accept connections
  - request/reply for commands
  - stream events (chat deltas, status/config changes)
- [ ] Integration tests for WS protocol (connect, ping, get status)

### 4) Wire into daemon
- [ ] Add flags/config for WS bind address + port (default localhost)
- [ ] Start WS server in daemon main
- [ ] Ensure clean shutdown

### 5) Config endpoints parity
- [ ] Define typed config struct (~10 settings)
- [ ] Implement `GetConfig`/`SetConfig`
- [ ] Emit `ConfigChanged`

### 6) Chat streaming parity
- [ ] Implement `SendMessage` streaming (`MessageDelta`, `MessageCompleted`)
- [ ] Backpressure + cancellation

## Progress log

### 2026-02-25
- Created branch `feature/ws-api`.
- Added docs: `docs/API_TRANSPORT.md`.
- Created this progress tracker.
