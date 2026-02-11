# Desktop Assistant — Development Guidelines

## Project Overview

A modular desktop assistant that exposes its API over D-Bus, runs as a systemd user service, and can interact with both user-level and system-level D-Bus services.

## Architecture

**Hexagonal (Ports & Adapters) Architecture**

```
                  ┌─────────────────────────┐
   D-Bus API ───► │      Inbound Ports       │
                  │   (trait interfaces)      │
                  ├─────────────────────────┤
                  │      Core Domain         │
                  │   (business logic,       │
                  │    pure functions)        │
                  ├─────────────────────────┤
                  │     Outbound Ports       │
   System D-Bus,  │   (trait interfaces)      │
   Filesystem ◄── │                           │
                  └─────────────────────────┘
```

- **Inbound ports**: Trait-defined interfaces that the D-Bus adapter calls into.
- **Core domain**: Business logic with no knowledge of D-Bus, filesystem, or other infrastructure.
- **Outbound ports**: Trait-defined interfaces the core uses; adapters implement them for D-Bus, filesystem, etc.

## Development Rules

### 1. Test-Driven Development (TDD)

- Write tests **before** writing the implementation.
- Tests define the expected behavior of each function.
- When a test fails, determine whether the **test** or the **code** is wrong — do not blindly adjust tests to pass.
- Cover edge cases explicitly.
- Write both **unit tests** (per module) and **integration tests** (`tests/` directory).

### 2. D-Bus API

- The primary external API is D-Bus.
- The service registers on the **user-level session bus**.
- It must be able to call services on the **system bus** when needed.
- D-Bus interface definitions should be separated from business logic (ports & adapters).

### 3. Systemd & XDG

- Designed to run as a **systemd user service** (`systemctl --user`).
- Follow XDG Base Directory Specification:
  - `$XDG_CONFIG_HOME` — configuration files
  - `$XDG_DATA_HOME` — persistent data
  - `$XDG_STATE_HOME` — logs, state
  - `$XDG_CACHE_HOME` — caches
  - `$XDG_RUNTIME_DIR` — runtime sockets/files

### 4. Code Quality

- **Modular and concise** — no unnecessary abstractions, no dead code.
- **No duplication** — before writing new code, check for existing logic that can be reused or extended.
- **Async-first** — use `async`/`await` with tokio for I/O-bound work.
- **Traits for abstraction** — define behavior through traits; implement adapters separately.
- **`From`/`Into` traits** — use for all data type conversions between layers.
- **Format and test after every change** — run `cargo fmt`, `cargo test`, then commit.

### 5. Task Tracking

- Each task is a markdown file in the `tasks/` directory.
- Task files must include:
  - **Background**: Why the change is needed.
  - **Change Description**: What will be modified or added.
  - **Expected Behavior**: How the system should behave after the change.
- Maintain `tasks/PRIORITY.md` as an ordered queue of tasks to process.

### 6. Commit Discipline

After every set of changes:
1. `cargo fmt`
2. `cargo test`
3. Commit with a clear message describing the change.

## Project Structure

```
desktop-assistant/
├── Cargo.toml                 # Workspace root
├── AGENTS.md                  # This file
├── tasks/                     # Task tracking
│   └── PRIORITY.md            # Ordered task queue
├── crates/
│   ├── core/                  # Domain logic (no I/O dependencies)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   └── ports/         # Trait definitions (inbound + outbound)
│   │   └── Cargo.toml
│   ├── dbus-interface/        # D-Bus adapter (inbound port implementation)
│   │   ├── src/
│   │   └── Cargo.toml
│   └── daemon/                # Main binary — wires everything together
│       ├── src/
│       │   └── main.rs
│       ├── tests/             # Integration tests
│       └── Cargo.toml
```

## Rust Conventions

- Edition: 2024
- Async runtime: tokio
- D-Bus library: zbus
- Error handling: thiserror for library errors, anyhow for application errors
- Serialization: serde where needed
- Logging: tracing
