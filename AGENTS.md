# Agent Instructions

## Dependency Security Policy

After adding any new packages, **scan for CVEs before building**.

Build scripts (e.g. `build.rs` in Rust, install scripts in npm) execute at build time and are a potential attack vector. Scanning the updated lockfile *before* running a build catches malicious or vulnerable transitive dependencies before any build-time code can execute.

### Workflow

1. Add the dependency (e.g. `cargo add <crate>`) — this updates the lockfile but does not build.
2. Scan immediately using `cve-mcp scan_packages` — parse the updated lockfile and pass all (name, version, ecosystem) tuples to the tool.
3. Review findings. Investigate any Critical or High severity issues before proceeding.
4. Only build once the scan is clean (or findings are understood and accepted).

This applies regardless of ecosystem (Cargo, npm, PyPI, etc.).

## Rust Conventions

Apply these consistently across the workspace. The pre-commit checklist at the bottom is the floor; project-specific patterns are anchored to existing examples in the codebase rather than abstract rules.

### Coding
- `?` for error propagation. Reserve `unwrap` / `expect` for tests and proven invariants. When `expect`ing in production, the message must explain the invariant — not just describe what would be unwrapped.
- Prefer `&str` / `&[T]` in argument position; take ownership only when storing.
- Newtype wrappers for invariant-bearing values. Examples here: `ConnectionId`, `ModelRef`, `ConnectionRef`.
- `From` / `Into` for type conversions; don't write `to_*` methods when traits suffice.
- Combinators (`map`, `and_then`, `unwrap_or_else`, `?`) over `match` for short `Option` / `Result` chains. Use `match` when there's branching control flow with side effects.
- Avoid `.clone()` on hot paths. `Arc<T>` for shared immutable, `Arc<Mutex<T>>` / `Arc<RwLock<T>>` for shared mutable.

### `unsafe`
- Don't use `unsafe` unless it's necessary AND you've reasoned about soundness. The bar is high.
- Required cases here: `std::env::set_var` / `remove_var` (Rust 2024 edition makes these `unsafe` because libc env-mutation is not threadsafe). Anything else needs a strong justification.
- Every `unsafe` block must have a `// SAFETY:` comment naming the invariant the caller is relying on. No "obvious" unsafe — write the soundness argument down. Example from `crates/daemon/src/registry.rs`:

  ```rust
  // SAFETY: single-threaded test; unique env-var name; no other code touches it.
  unsafe { std::env::remove_var(&unused); }
  ```

### Testing
- Unit tests colocated as `#[cfg(test)] mod tests {}` in lib files.
- Integration tests in `tests/` next to `Cargo.toml`.
- `#[tokio::test]` for async; `#[tokio::test(flavor = "multi_thread")]` only when explicitly testing concurrent behavior.
- Mock at trait boundaries. For HTTP: `httpmock` (already a daemon dev-dep). For time: an injected `Clock` trait — see `BedrockClient::ModelClock` in `crates/llm-bedrock`.
- Determinism: sort outputs before assertion; never depend on hash iteration order.
- `expect("descriptive reason")` over `unwrap()` in tests so failure messages are self-explanatory.
- Test public behavior, not private implementation. If a private fn needs testing, surface as `pub(crate)` with a documented contract.
- Don't hold `std::sync::MutexGuard` across `.await`. Drop the guard explicitly before awaiting — `clippy::await_holding_lock` flags this.

### Generics
- `impl Trait` in argument position for single-bound, single-use parameters.
- Named generics with `where` clauses for multiple bounds, recursion, or readability.
- Avoid generic explosion: 3+ generic parameters usually indicates a missing struct or associated type.
- Prefer `Arc<dyn Trait>` over hand-rolled enum-dispatch when there are many implementors and no perf-critical specialization (see open issue #44 for the `AnyLlmClient` cleanup).
- Trait bounds: keep `Send + Sync + 'static` co-located on the trait def when the trait is only useful in async contexts.

### Error handling
- Library crates: `thiserror` with structured variants (e.g. `core::CoreError`).
- Binary crates: `anyhow` with `Context::context()` for narrative.
- **Never pattern-match on error message strings.** Pattern-match on variants. If you find yourself doing `error.to_string().contains("429")`, the upstream type is throwing away structured info that should be preserved (see open issue #46).
- Surface enough context in `Display` for debugging without leaking secrets — see `redacted_secret_audit` for the API-key fingerprint pattern.

### Async
- Don't hold non-async locks (`std::sync::Mutex`, `parking_lot::Mutex`) across `.await`. Drop the guard explicitly, or use `tokio::sync::Mutex` if the lock genuinely needs to span the await.
- `tokio::join!` for independent parallel work; `tokio::try_join!` when both must succeed and the first error should cancel the rest.
- Long-running spawned tasks need cancellation — channel-based or `CancellationToken`. Don't leak.
- Cross-cutting context: `tokio::task_local!`. Existing examples: `REASONING_CONFIG`, `MODEL_OVERRIDE`, `CONTEXT_BUDGET` in `crates/core/src/ports/llm.rs`, plus `ACTIVE_CLIENT` in `crates/daemon/src/routing_llm.rs`.

### Workspace organization
- Hexagonal layout: trait boundaries in `core::ports`, infrastructure adapters in `daemon` / `storage` / `llm-*`, wire types in `api-model`.
- LLM provider crates (`crates/llm-*`) MUST NOT depend on the `daemon` crate. Cross-cutting context flows via task-locals defined in `core`.
- Wire types (`api-model`) separate from domain types (`core::domain`). The daemon's mapper layer translates between them.

### Documentation
- Doc comments (`///`) on every public item.
- Include rationale (`Why:` lines) for non-obvious choices, not just descriptions of behavior.
- For shared trait-locks / task-locals, document the contract in the module-level doc.
- Don't narrate PR / issue history in code comments. Reference issues only when the comment captures a non-obvious WHY tied to that issue.

### Pre-commit checklist
1. `cargo clippy --workspace --all-targets`
2. `cargo test --workspace`
3. (Future: `cargo fmt --check` and `cargo clippy ... -- -D warnings` once the pre-existing warnings are remediated — tracked in follow-up issues.)
