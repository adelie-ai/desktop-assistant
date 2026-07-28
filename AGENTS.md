# Agent Instructions — desktop-assistant

Shared standards live in [AGENTS.base.md](AGENTS.base.md), which is generated. This file holds the rules specific to this repo.

Repo-specific conventions for the Adelie daemon and its workspace crates. This file covers what is specific to *this* codebase. The overrides and additions to the base are listed at the end.

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
- Prefer `Arc<dyn Trait>` over hand-rolled enum-dispatch when there are many implementors and no perf-critical specialization. A hand-rolled `AnyX` enum re-dispatches every trait method by hand and grows a variant per implementor, so it collapses once the set is open-ended. Storage and LLM calls already await on a socket; a vtable hop is not the cost that matters.
- `Send + Sync + 'static` co-located on the trait def when the trait is only useful in async contexts.

### Error handling
- Library crates: `thiserror` with structured variants (e.g. `core::CoreError`).
- Binary crates: `anyhow` with `Context::context()` for narrative.
- **Never pattern-match on error message strings.** Pattern-match on variants. `error.to_string().contains("429")` means the upstream type is throwing away structured info that should be preserved. Fix the upstream type to carry the distinction as a variant instead of parsing its `Display` output.
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
- A new `.sql` file is only run once it is registered in the `MIGRATIONS` list in `crates/storage/src/pool.rs` (`migration!("039_thing.sql")`, in ordinal order). The runner applies each entry **at most once**, recording its file name in the `schema_migrations` ledger, and serializes the whole run behind a `pg_advisory_lock` so two daemons booting against one database queue instead of racing.
- Consequences of that ledger, all of them load-bearing:
  - **Never edit or rename a migration that has shipped.** Databases that recorded it will not run it again, so the edit reaches only fresh installs and the two silently diverge. Fix a shipped migration with a new one.
  - **Every migration must still be idempotent.** A database migrated before the ledger existed has no record of anything, so its first boot under this runner replays the whole set once.
  - **A migration is applied and recorded in one transaction**, so nothing that cannot run inside a transaction block (`CREATE INDEX CONCURRENTLY`, `VACUUM`) belongs in one.
  - Behaviour that used to come free from re-running every file on every boot no longer does. Migration 029's RLS policy list is the example: a later migration that adds a user-scoped table must enable RLS itself.
- Schema changes that touch personal-data tables must respect the multi-tenant boundary: the table carries `user_id`, every query is scoped by it, and the migration that creates the table enables its RLS policy in the same migration.

## Daemon entry points & operations

The daemon is built and run as `cargo run -p desktop-assistant-daemon`. Operational recipes — installing as a systemd user service, running a parallel dev instance, packaging — live in the `justfile` and `README.md`. When adding new operational behavior, prefer extending an existing `just` recipe over inventing a new entry point.

## Build hygiene

The workspace is held to:

- a RustSec advisory scan of `Cargo.lock` (`scripts/audit.sh`)
- a working-tree secret scan (`scripts/secret-scan.sh`) - see "Secret scanning" below
- `cargo fmt`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- the same `clippy` and `test` runs for `crates/storage-sqlite` with its
  `--features sqlite` on (`just lint-sqlite`, `just test-sqlite`)

The last pair is not redundant. That crate is entirely behind an off-by-default
`sqlite` feature - the feature exists so the daemon build never links the
sqlite C library - so the `--workspace` steps compile an empty shell and run
none of its tests. Only a step that names the crate *and* the feature covers
it, and the shell suite `scripts/tests/sqlite-gate.test.sh` holds the gate to
having one.

`just check` runs all of those, each workspace step followed by its
`storage-sqlite` counterpart (`lint` then `lint-sqlite`, `test` then
`test-sqlite`), with `build` in between and `just test-scripts` last (the named
tests for the gate's own shell steps, under `scripts/tests/`). The advisory scan
and the secret scan both come first, before either `lint` or `build` - build
scripts execute at first compile, under `clippy` as much as under `build`, and
enabling `sqlite` adds `libsqlite3-sys`, which compiles C.

What `just check` does **not** cover:

- The DB-gated `crates/storage` isolation suites. They pass-skip without
  `TEST_DATABASE_URL`, so a green `just check` proves nothing about
  multi-tenant safety - run `just test-db` for that, and say in the PR which of
  the two you ran.
- The parallel-safety of `just test-db` itself, unless a container runtime is
  reachable. That criterion provisions two real databases; with no
  podman/docker it skips, loudly and by name, so a green `just check` on a
  machine without one has not verified it.

So the gate wants a reachable podman/docker and the network - the advisory scan
fetches, and runs offline only under the `ADELE_AUDIT_ALLOW_STALE=1` opt-in
described below. The secret scan adds no network requirement: gitleaks' rule
set is bundled in the binary, not fetched per run. That goes for `git push`
too: the pre-push hook runs the same `just check`.

New code keeps it there. Warnings-as-errors is enforced **mechanically**, not by reviewer vigilance: the root `Cargo.toml` sets `[workspace.lints] rust.warnings = "deny"` and `clippy.all = "deny"`, and every member inherits via `[lints] workspace = true`, so a plain `cargo build`/`test`/`clippy` hard-fails on any warning. Base rule 2.1 states the posture.

## Dependency safety

Common, well-maintained cargo plugins are fine — `cargo-edit` (`cargo upgrade`/`add`/`rm`), `cargo-audit`, `cargo-outdated`, `cargo-deny`; prefer built-in cargo for trivial one-line edits, and avoid obscure/unmaintained single-author plugins without checking first. The `cve-mcp` MCP server's `scan_packages` tool is wired up; base rule 6.1 covers when and how to use it, and the 6.1 entry at the end of this file gives the group's scan steps. Repo-specific note: build scripts (`build.rs`) execute on first build, so the scan happens between lockfile change and first `cargo build`, not after.

`cargo-audit` is therefore a prerequisite of the gate, not an optional extra:
`cargo install cargo-audit --locked`. The scan step (`just audit`,
`scripts/audit.sh`) is deliberately unable to pass by accident:

- **cargo-audit not installed** - hard failure, with the install command.
- **Advisory database unfetchable** (offline, blocked proxy) - hard failure. It
  will not fall back to a cached database silently. If you accept a possibly
  stale database, opt in with `ADELE_AUDIT_ALLOW_STALE=1`; the run then prints a
  loud banner naming the database and its age, and you say so in the PR.
- **Vulnerability reported** - hard failure (base rule 6.1: a high or critical advisory blocks the change).
- **Informational advisory** (unmaintained, unsound, yanked) - reported loudly,
  does not fail the gate; treat it as a review item.
- **Advisory suppressed by an audit.toml `ignore` list** - named in the output
  and in the summary line, and does not fail the gate: suppressing one is a
  reviewed decision, but the gate never presents it as a clean scan.

## Secret scanning

`just secret-scan` (`scripts/secret-scan.sh`) runs [gitleaks](https://github.com/gitleaks/gitleaks)
over the **checked-out files**, not git history. This is deliberate, not an
oversight: #811's key was never committed, so a history-based scan
(`gitleaks git`, or the older `gitleaks detect`) would have reported clean for
the entire time the key was exposed. The script calls `gitleaks dir`, the
filesystem-walk subcommand, which reads whatever is on disk - tracked or not,
gitignored or not.

`gitleaks` is therefore a prerequisite of the gate, the same as `cargo-audit`:
install the pinned version (`pacman -S gitleaks` on Arch/CachyOS, or
`brew install gitleaks`). The gitleaks rule set is bundled in the binary
itself, not fetched per run, so the version is pinned in
`scripts/secret-scan.sh` (`GITLEAKS_VERSION`) rather than left to whatever a
machine happens to have - two developers running two different rule sets is
the same reproducibility failure as two different Rust toolchains. Bump the
pin deliberately, after checking the new release's notes (base rule 6.1), not
by installing whatever a package manager offers this week.

Two files at the repo root configure the scan:

- **`.gitleaks.toml`** extends (not replaces) gitleaks' default rule set and
  allowlists a short list of generated/vendored directories - `target/`,
  `build/`, `.git/`, `.venv/`, `.claude/worktrees/` - that are never
  hand-authored and only slow the scan down. `.flatpak-builder/` is
  deliberately **not** on that list: it is exactly where the `.env` copies in
  #811 were found, so the scanner must still be able to see it.
- **`.gitleaksignore`** lists reviewed false positives by gitleaks fingerprint
  (`file:rule-id:start-line`), one entry per finding, each with a comment
  explaining why it is not a real secret (this repo's own test-only PEM
  fixtures and a couple of truncated example tokens in docs, at the time of
  writing). Because the fingerprint pins the line number too, an unrelated
  edit that shifts the match makes the finding reappear rather than silently
  keeping a stale exemption - re-add it once you have re-confirmed it is still
  a fixture, not a regression.

Like the advisory scan, the step is deliberately unable to pass by accident:

- **gitleaks not installed** - hard failure, with the install command.
- **Installed version does not match the pin** - hard failure, naming both
  versions.
- **A scan that produced no report** - hard failure. An exit status alone is
  not proof a scan ran (the same #706 failure mode `scripts/audit.sh` guards
  against): a fatal gitleaks error (bad config, bad path) exits non-zero and
  writes no report at all, so report-existence, not exit code alone, is what
  the script trusts.
- **A secret found** - hard failure, always. There is no suppress-and-warn
  path for a real finding the way there is for an informational RustSec
  advisory; add a reviewed `.gitleaksignore` entry for a genuine false
  positive, or rotate and remove a real one.

## Overrides and additions to the shared base

Everything in [AGENTS.base.md](AGENTS.base.md) applies to this repo. This section
records only the points where this repo deliberately differs from the base, or adds a
rule the base does not have.

### 3.1 The gate for this repo (addition)

The `adelie-ai` repos have no CI. The gate is local and the author runs it: `just check`.
**Build hygiene** above states exactly what that covers and what it does not, including the
DB-gated suites that `just check` cannot prove. Say in the pull request which of `just check`
and `just test-db` you ran. Run `just install-hooks` once per clone to put the same gate on
pre-push.

### 4.3 Branch and pull request - merge when green (override, weaker than the base)

The base opens a pull request and waits for the user. In these repos the merge is delegated:
merge your own pull request as soon as it is green and independently shippable. Green here
means more than a clean build. The gate above passed, the tests cover the new behavior and
not only the absence of a panic, the security pass is done, and the change stands on its own.
Assign `dspadea` with `gh pr edit --add-assignee` and verify it; a review request from the
same account no-ops without an error, so never report a pull request as review-requested.
When in doubt, hold.

### 4.4 Worktrees - the group convention (addition)

Put the worktree at `.worktrees/<repo>/issue-N-slug/` under the group directory, on a branch
that mirrors the slug. Before you run tasks in parallel worktrees, look for shared files,
shared `Cargo.toml` dependency edits, and shared migration ordinals. Serialize the work where
they overlap, and tell each parallel agent the scope it owns.

### 6.1 Dependencies - the group's scan workflow (addition)

Base rule 6.1 sets the policy, including that a high or critical advisory blocks the change.
This group runs it with its own tooling:

1. Add the dependency (`cargo add <crate>`). This writes the lockfile but does not build.
2. Scan the updated lockfile with the `cve-mcp` server's `scan_packages` tool, or with
   `cargo audit`. Pass every (name, version, ecosystem) tuple.
3. Build only after the scan is clean, or after you have accepted the findings in writing.

### 6.2 Local parity - a shared resource must survive concurrent sessions (addition)

Several sessions run at the same time. `just test-db` gives every invocation its own
container and its own host port, and removes only what it created, so two sessions can run
the storage suites at once. Hold any new recipe that provisions something shared to
the same bar. A fixed name or a fixed port turns a second session into a plausible-looking
flake in an unrelated test.

### 7.3 Earn abstractions - reuse the ports-and-adapters layout (addition)

Reuse the existing traits and the ports-and-adapters layout rather than inventing a parallel
one. Extend an existing crate instead of adding one, unless the seam is obvious.

### 9.1 Tracker for this project

GitHub Issues on `github.com/adelie-ai/desktop-assistant`, together with the shared `adelie-ai` project
board. Manage entries with the `gh` CLI (`gh issue create`, `gh issue list`, `gh issue edit`,
`gh pr create`). The board states in use are In Progress, In Review, and Done.

### Platform, not a single product (addition)

Adele is a platform, not one product. Solve for the general case at every seam that is
plural by domain: storage backends, LLM providers, transports, clients, MCP servers, speech
engines. When a requirement names two of something, ask whether the real requirement is N
of them, and build that one instead.

Put the abstraction at the port. Keep the conditional compilation and the selection in one
factory, so a new implementation costs a crate, a feature, and one arm - not an edit to
every implementation that already exists. A hand-rolled `AnyX` enum with a variant per
implementation is the shape that fails this test: it re-dispatches every trait method by
hand and grows with the set.

Base rule 7.3 still holds inside a component. Do not invent indirection that a single call
site does not need. It does not licence the narrow build at a platform seam, because there
the plurality is the product, and the seam is already past the three-call-site test.

Fail loudly and by name when a configured selection is not compiled in, or is unavailable.
Name what was asked for and what is actually present. Silent degradation to a lesser
backend hides the problem from the one person who could fix it.

### Capability-based degradation (addition)

Every reliance on an optional operating-system or desktop service - logind, the screen lock,
KDE and Plasma, PipeWire specifics, any session-bus or system-bus D-Bus interface - must be
capability-detected, and must degrade cleanly when the service is absent. Never make one a
hard dependency that errors or hangs. The product can run headless, in a container, on
another desktop environment, or as a system service.

Distinguish "is the capability present?" from "did my call succeed?". There are three states.
Absent: disable that feature, log once, and fall back to the prior behavior. Present and
known: use it. Present but anomalous: stay conservative, or hold the last known state, and
warn. Scope any privacy or safety fail-safe to the last two states only. A fail-safe that is
correct on the desktop can be pathological headless. "Treat an unknown session as inactive"
means the microphone never opens.

Detect each optional dependency on its own. The absence of one never disables the others and
never aborts startup. Surface the detected capability, so an operator can see why a feature
is on or off.
