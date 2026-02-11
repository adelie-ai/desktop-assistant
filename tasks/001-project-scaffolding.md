# 001 — Project Scaffolding

## Background

The desktop assistant needs a workspace structure that supports hexagonal architecture with clear separation between domain logic, D-Bus interface, and the daemon binary. This is the foundation all future work builds on.

## Change Description

- Set up a Cargo workspace with three crates:
  - `crates/core` — domain logic and port trait definitions, no I/O dependencies.
  - `crates/dbus-interface` — D-Bus adapter implementing inbound ports via zbus.
  - `crates/daemon` — binary crate that wires adapters to the core and starts the service.
- Add initial dependencies (tokio, zbus, thiserror, anyhow, tracing, serde).
- Create placeholder `lib.rs` / `main.rs` files with module stubs.
- Add a basic integration test that asserts the workspace compiles and the daemon binary entry point is reachable.

## Expected Behavior

- `cargo build` succeeds across the entire workspace.
- `cargo test` passes with at least one trivial test per crate.
- The crate dependency graph enforces the architectural boundary: `core` depends on nothing project-internal, `dbus-interface` depends on `core`, `daemon` depends on both.
