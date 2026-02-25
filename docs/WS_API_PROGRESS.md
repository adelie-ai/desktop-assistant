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
- [ ] Implement `SendMessage` streaming (`AssistantDelta`, `AssistantCompleted`)
- [ ] Backpressure + cancellation

## Progress log

### 2026-02-25
- Surveyed existing D-Bus APIs:
  - `org.desktopAssistant.Conversations`: create/list/get/delete/clear, plus `send_prompt` streaming via signals
  - `org.desktopAssistant.Settings`: LLM + embeddings + persistence settings, write-only API key
- Added `crates/api-model` with canonical `Command` / `CommandResult` / `Event` types and view structs.
  - Naming: `SendMessage` (not `SendPrompt`), streaming events `AssistantDelta`/`AssistantCompleted`/`AssistantError`.

- Created branch `feature/ws-api`.
- Added docs: `docs/API_TRANSPORT.md`.
- Created this progress tracker.
