# Agent Instructions — desktop-assistant

Repo-specific conventions for the Adelie daemon and its workspace crates. Cross-project engineering standards that apply to every `adelie-ai` repo (don't-break-`main`, spec-driven testing, warnings-are-failures, security review, maintainability, capability-based degradation, GitHub/board hygiene, worktrees) are embedded below under **Cross-project engineering standards** so a contributor working only in this repo has them in hand. The rest of this file covers what is specific to *this* codebase.

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

New code keeps it there. Warnings-as-errors is enforced **mechanically**, not by reviewer vigilance: the root `Cargo.toml` sets `[workspace.lints] rust.warnings = "deny"` and `clippy.all = "deny"`, and every member inherits via `[lints] workspace = true`, so a plain `cargo build`/`test`/`clippy` hard-fails on any warning. See the **Warnings are failures** standard below for the posture.

## Dependency safety

Common, well-maintained cargo plugins are fine — `cargo-edit` (`cargo upgrade`/`add`/`rm`), `cargo-audit`, `cargo-outdated`, `cargo-deny`; prefer built-in cargo for trivial one-line edits, and avoid obscure/unmaintained single-author plugins without checking first. The `cve-mcp` MCP server's `scan_packages` tool is wired up; the **Security review** standard below covers when and how to use it. Repo-specific note: build scripts (`build.rs`) execute on first build, so the scan happens between lockfile change and first `cargo build`, not after.

## Cross-project engineering standards

These apply to every repo under `github.com/adelie-ai`. They're embedded in each repo's `AGENTS.md` (not centralized) so a contributor working in a single repo has them in hand. Operator-specific preferences and machine-specific deploy recipes are intentionally not here.

### Don't break `main`
- `main` is the release: at any commit it must build, test, and run.
- Merge a green change as soon as it's independently shippable — additive, behavior-preserving, or behind a default that preserves the old path. Don't hold green work hostage to a coordinated release.
- Co-dependent changes land together; name the interlock ("blocked-by #X" / "must merge with #Y") so it's visible without reading the diff.
- "Green" is more than CI: review passed, tests cover the new behavior (not just "no panic"), warnings clean, security pass done, change stands on its own. With no active CI in these repos, "green" rests on local `cargo test --workspace` + `fmt` + `clippy` + `cargo audit` (`just check`), run by the author.
- When in doubt, hold. A half-coupled "fix-forward" merge breaks `main` for everyone.

### Tests are spec-driven (TDD)
- Every change carries a Testing section: acceptance criteria as testable assertions, each criterion a named test whose name is legible from test output.
- Write failing tests first, in their own commit before the implementation commit — that commit is the spec.
- Cover all new code: every branch, error path, edge case. Gaps are a review finding.
- Assert the desired outcome, not just that a call returned `Ok`.
- Enumerate unhappy paths deliberately: empty/missing input, boundary/max, concurrent/racy, authorization/tenant boundaries, partial reads/writes/dropped streams, malformed input. A test list with none of these is testing wishes.

### Warnings are failures
- Compiler warnings, clippy lints, formatter diffs, and advisories all count — fix the root cause. If a lint truly doesn't apply, suppress at the narrowest scope with a one-line justification; never crate-wide.
- This repo enforces it **mechanically**: the workspace `[lints]` table denies `rust.warnings` and `clippy.all`, so `cargo build`/`test`/`clippy` hard-fail on a warning — it isn't left to reviewer attention.
- Never `--no-verify` past hooks. If a hook is genuinely broken, fix it in its own commit and explain why.
- Don't `#[ignore]` a test you broke; fix it, or open a tracking issue and reference it from the attribute.
- Pre-existing warnings in a file you touch are yours to address (in-change or a small follow-up) — don't pile new code on an ignored signal.

### Security review before requesting review
- Read your own diff adversarially: untrusted input crossing trust boundaries (network, IPC, D-Bus, MCP tool args), secrets in logs, missing auth checks, panic-on-input, unparameterized SQL/shell.
- Scan dependencies whenever the lockfile changed (`cargo audit` or the `cve-mcp` server) — and scan BEFORE the first build, because build scripts execute attacker-controlled code at build time.
- High/critical CVEs are hard blockers: patch in the same change, prove the path unreachable and document why, or file a tracked follow-up referenced in the change. Never ship past one silently; never pin around an advisory without a comment or tracking issue.

### Maintainability / cognitive load
- Keep each change small enough to land independently with a clear deliverable.
- Don't introduce a new abstraction until ~3 call sites prove the pattern; when one new type unifies several needs, justify the unification explicitly.
- Reuse existing traits and the ports-and-adapters layout rather than inventing parallel ones; extend an existing crate over adding one unless the seam is obvious.

### Capability-based degradation
- Every reliance on an optional OS/desktop service (logind, screen-lock, KDE/Plasma, PipeWire specifics, any session- or system-bus D-Bus interface) must be capability-detected and degrade gracefully — never a hard dependency that errors or hangs when absent. The product may run headless, in containers, on other DEs, or as a system service.
- Distinguish "is the capability present?" from "did my call succeed?" Three states: absent → disable that feature, log once, fall back to prior behavior; present-and-known → use it; present-but-anomalous → stay conservative / last-known-state and warn. Scope any privacy/safety fail-safe to the last two — a fail-safe correct on the desktop can be pathological headless (e.g. "treat unknown session as inactive" ⇒ mic never opens).
- Detect each optional dependency independently; absence of one never disables the others or aborts startup. Surface the detected capability so an operator sees *why* a feature is on or off.

### GitHub issue / PR / board hygiene
- Self-assign an issue when you start it (or comment to claim it) so parallel work doesn't collide; move the board card to In Progress.
- Link the PR to the issue: `Closes #N` to auto-close, `Refs #N` when it only partially addresses it.
- Keep the board in sync with reality (In Review on open, Done on merge); if you can't move the card, comment the intended status.
- On multi-session work, leave a short status comment before stopping — what landed, what's next, what's blocked — so state is reconstructable without git log.

### Worktrees
- Do code work in a git worktree on its own branch off `origin/main`, never the primary checkout, so concurrent sessions don't collide. Convention: `~/Projects/adelie-ai/.worktrees/<repo>/issue-N-slug/`, branch mirroring the slug.
- Run independent tasks in parallel worktrees, but check first for shared files / shared `Cargo.toml` dep edits / shared migration ordinals — if they overlap, serialize. Brief each parallel agent on its scope ("own crate X, don't touch Y").
