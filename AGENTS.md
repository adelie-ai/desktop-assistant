# Agent Instructions — desktop-assistant

Repo-specific conventions for the Adelie daemon and its workspace crates. Cross-project workflow rules (issue/PR/board sync, parallel worktrees, warnings-are-failures, security review posture, TDD posture) live in the user's memory and are not duplicated here — this file covers what is specific to *this* codebase.

## Workspace shape

Hexagonal layout. Trait boundaries live in `core::ports`; infrastructure adapters live in `daemon`, `storage`, and `llm-*`; wire types live in `api-model`. Cross-cutting context flows via task-locals defined in `core` (e.g. `REASONING_CONFIG`, `MODEL_OVERRIDE`, `CONTEXT_BUDGET` in `crates/core/src/ports/llm.rs`; `ACTIVE_CLIENT` in `crates/daemon/src/routing_llm.rs`).

- LLM provider crates (`crates/llm-*`) MUST NOT depend on the `daemon` crate. If you find yourself reaching for daemon state from a provider, it belongs on a task-local or a port trait.
- Wire types (`api-model`) are separate from domain types (`core::domain`). The daemon's mapper layer translates between them. Don't leak wire shapes into domain code or vice versa.
- Prefer extending an existing crate over adding a new one. New crates need an obvious seam — a stable trait boundary, a different dependency profile, or a different consumer.

## Rust conventions

### Coding
- `?` for error propagation. `unwrap` / `expect` are for tests and proven invariants. Production `expect` must explain the invariant, not just describe what is being unwrapped.
- `&str` / `&[T]` in argument position; take ownership only when storing.
- Newtype wrappers for invariant-bearing values (existing examples: `ConnectionId`, `ModelRef`, `ConnectionRef`).
- `From` / `Into` over `to_*` methods when traits suffice.
- Combinators (`map`, `and_then`, `unwrap_or_else`, `?`) for short `Option` / `Result` chains; `match` when there's branching with side effects.
- Avoid `.clone()` on hot paths. `Arc<T>` for shared immutable; `Arc<Mutex<T>>` / `Arc<RwLock<T>>` for shared mutable.

### `unsafe`
The bar is high and the soundness argument must be written down in a `// SAFETY:` comment naming the invariant the caller relies on. Don't ship "obvious" unsafe. The only currently-acceptable case is the Rust 2024 edition's `unsafe { std::env::set_var(...) }` / `remove_var` because libc env-mutation is not thread-safe; anything else needs explicit justification.

### Async
- Don't hold non-async locks (`std::sync::Mutex`, `parking_lot::Mutex`) across `.await`. Drop the guard explicitly, or use `tokio::sync::Mutex` if the lock genuinely needs to span the await. `clippy::await_holding_lock` flags this and is not a suggestion.
- `tokio::join!` for independent parallel work; `tokio::try_join!` when both must succeed and the first error should cancel the rest.
- Long-running spawned tasks need cancellation — channel or `CancellationToken`. Don't leak.
- Cross-cutting context propagates via `tokio::task_local!`. Don't add new ones casually; document the contract in the module-level doc when you do.

### Generics
- `impl Trait` in argument position for single-bound, single-use parameters.
- Named generics with `where` clauses for multiple bounds, recursion, or readability.
- 3+ generic parameters usually signals a missing struct or associated type.
- Prefer `Arc<dyn Trait>` over hand-rolled enum-dispatch when there are many implementors and no perf-critical specialization (see open issue #44 for the `AnyLlmClient` cleanup).
- `Send + Sync + 'static` co-located on the trait def when the trait is only useful in async contexts.

### Error handling
- Library crates: `thiserror` with structured variants (e.g. `core::CoreError`).
- Binary crates: `anyhow` with `Context::context()` for narrative.
- **Never pattern-match on error message strings.** Pattern-match on variants. `error.to_string().contains("429")` means the upstream type is throwing away structured info that should be preserved (open issue #46).
- `Display` should carry enough context for debugging without leaking secrets — see the `redacted_secret_audit` API-key fingerprint pattern.

### Testing
- Unit tests colocated as `#[cfg(test)] mod tests {}` in lib files; integration tests in `tests/` next to `Cargo.toml`.
- `#[tokio::test]` for async; `#[tokio::test(flavor = "multi_thread")]` only when explicitly testing concurrent behavior.
- Mock at trait boundaries. For HTTP: `httpmock` (already a daemon dev-dep). For time: an injected `Clock` trait — see `BedrockClient::ModelClock` in `crates/llm-bedrock`.
- Determinism: sort outputs before assertion; never depend on hash iteration order.
- `expect("descriptive reason")` over `unwrap()` in tests so failure messages are self-explanatory.
- Test public behavior, not private implementation. If a private fn needs testing, surface it as `pub(crate)` with a documented contract.

### Documentation
- Doc comments (`///`) on every public item.
- Include rationale (`Why:` lines) for non-obvious choices, not just descriptions of behavior.
- For shared trait-locks / task-locals, document the contract in the module-level doc.
- Don't narrate PR / issue history in code comments. Reference issues only when the comment captures a non-obvious WHY tied to that issue.

## Storage & migrations

- Migrations are append-only and ordinally numbered. Two concurrent PRs cannot share an ordinal — coordinate before opening, or rebase to take the next number. This is the one place parallel worktrees genuinely conflict; the conflict is invisible until both PRs merge, so check before pushing.
- Schema changes that touch personal-data tables must respect the multi-tenant boundary (`user_id` scoping). See the multi-tenant schema work (#102) as the reference shape.

## Daemon entry points & operations

The daemon is built and run as `cargo run -p desktop-assistant-daemon`. Operational recipes — installing as a systemd user service, running a parallel dev instance, packaging — live in the `justfile` and `README.md`. When adding new operational behavior, prefer extending an existing `just` recipe over inventing a new entry point.

## Build hygiene

The workspace is held to:

- `cargo fmt`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

New code keeps it there. The user-memory "warnings are failures" rule covers the posture; this note exists so contributors know the baseline this repo enforces.

## Dependency safety

This workspace uses `cargo` exclusively for dependency management (no `cargo-edit`). The `cve-mcp` MCP server's `scan_packages` tool is wired up; the user-memory security-review rule covers when and how to use it. Repo-specific note: build scripts (`build.rs`) execute on first build, so the scan happens between lockfile change and first `cargo build`, not after.
