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
- Intra-doc links are checked by the gate (`just doc`), so a bracketed link must resolve to something a reader of the public docs can follow. When the target is deliberately private, write a plain code span instead of a link - never make an item public, and never add an `#[allow(rustdoc::...)]`, to satisfy a link.
- A module documented with its own `//!` header takes no `///` summary on its `mod` declaration. The two merge, and the merged text resolves its links against the PARENT module, which silently breaks every unqualified link in the header.

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
- `cargo doc --workspace --no-deps` (`scripts/doc.sh`) - see "Documentation lints" below
- `cargo test --workspace`
- the same `clippy`, `doc` and `test` runs for `crates/storage-sqlite` with its
  `--features sqlite` on (`just lint-sqlite`, `just doc-sqlite`,
  `just test-sqlite`)
- the same `clippy`, `doc` and `test` runs for `crates/client-common` with its
  `--features mcp-host` on (`just lint-mcp-host`, `just doc-mcp-host`,
  `just test-mcp-host`)

Neither extra set is redundant, for the same reason. `storage-sqlite` is
entirely behind an off-by-default `sqlite` feature - the feature exists so the
daemon build never links the sqlite C library. `client-common`'s
`mcp_host` module (the client-side MCP host: the spawn path a real desktop
session uses, as opposed to the daemon's own fleet) is behind an off-by-default
`mcp-host` feature, and - unlike `sqlite` - no other workspace crate enables it
as a normal dependency either, so nothing pulls it in via feature unification.
Either way the `--workspace` steps compile an empty shell and run none of that
code's tests. Only a step that names the crate *and* the feature covers it,
and a shell suite holds the gate to having one:
`scripts/tests/sqlite-gate.test.sh` and `scripts/tests/mcp-host-gate.test.sh`.

`just check` runs all of those, each workspace step followed by its
`storage-sqlite` and `mcp-host` counterparts (`lint` then `lint-sqlite` then
`lint-mcp-host`, `doc` then `doc-sqlite` then `doc-mcp-host`, `test` then
`test-sqlite` then `test-mcp-host`), with `build` in between and
`just test-scripts` last (the named tests for the gate's own shell steps, under
`scripts/tests/`). The advisory scan and the secret scan both come first,
before `lint`, `doc` or `build` - build scripts execute at first compile, under
`clippy` and `doc` as much as under `build`, and enabling `sqlite` adds
`libsqlite3-sys`, which compiles C.

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

## Documentation lints

`just doc` (`scripts/doc.sh`) runs `cargo doc --no-deps` over the workspace, plus
the two feature-gated repeats. It is the only step in the gate that evaluates
rustdoc lints: `cargo clippy -- -D warnings` reaches compiler lints only, so
without this step a doc link to a renamed, privatised or deleted item passes
everything else. Those lints become errors the same mechanical way the compiler
ones do - `[workspace.lints] rust.warnings = "deny"` reaches rustdoc too.

Like the two scans, the step is unable to pass by accident:

- **A rustdoc error** - hard failure, with rustdoc's own output. Fix the doc
  comment, and read the prose beside it: a link to an item that no longer exists
  usually means the sentence around it is stale too.
- **A rustdoc *warning*** - hard failure. A warning where every other crate gets
  an error means that crate is not inheriting `[lints] workspace = true`, so its
  documentation would rot behind a green gate.
- **A run that produced no documentation** - hard failure. `cargo doc` that
  selects nothing exits 0 and prints almost nothing, which reads in the log like
  "the docs are fine", so the step checks each selected crate's `index.html` on
  disk instead of trusting the exit status.

`scripts/tests/doc-gate.test.sh` holds the gate to having the step, and proves
the detection against throwaway fixture workspaces.

Do not silence a finding with `#[allow(rustdoc::...)]`, and do not widen an
item's visibility to satisfy a link. Where the target is genuinely private,
the link becomes a plain `code span`.

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

`gitleaks` is therefore a prerequisite of the gate, the same as `cargo-audit`.
**Install the pinned release tarball, user-local, no sudo:**

```sh
curl -sL -o gitleaks.tar.gz https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
tar -xzf gitleaks.tar.gz gitleaks && install -m 755 gitleaks ~/.local/bin/gitleaks
```

That is the primary path, not a fallback: `pacman -S gitleaks` also installs
the pinned `8.30.1`, but the CachyOS/Arch package (confirmed on both plain
Arch and CachyOS) does not set the build-time version ldflag, so
`gitleaks version` prints `version is set by build process` instead of a
version number - the exact pinned release then fails the pin check below.
This is a packaging defect in that distro build, not something this repo can
fix from here; the tarball release does set the ldflag correctly.

The gitleaks rule set is bundled in the binary itself, not fetched per run,
so the version is pinned in `scripts/secret-scan.sh` (`GITLEAKS_VERSION`)
rather than left to whatever a machine happens to have - two developers
running two different rule sets is the same reproducibility failure as two
different Rust toolchains. Bump the pin deliberately, after checking the new
release's notes (base rule 6.1), not by installing whatever a package
manager offers this week.

The scan runs in **two layers**, because this repository is public.

- **Layer 1 - the committed rules**, in `.gitleaks.toml`. Credentials, plus a
  second class the gate also has to catch: **private information**. None of
  that class is a credential, which is why an absolute home path, a private
  hostname and a live instance name all passed this gate for as long as they
  sat in the tree. They arrive the same way each time - somebody illustrates a
  point with real output from the running system, and the output carries the
  machine it came from. Layer 1 matches **shapes only**: an absolute home path
  (`/home/<name>`, `/Users/<name>`) and a hostname on a pseudo-TLD that public
  DNS never delegates (`.lab`, `.lan`, `.corp`, `.home`). No site-specific
  value belongs in this file - it is published with the repository, so a value
  written here is the leak, permanently, in git history.
- **Layer 2 - the host-local rules**, at
  `${XDG_CONFIG_HOME:-$HOME/.config}/adelie-ai/gitleaks-private.toml`, outside
  any repository and never committed. This is where the site-specific
  **literals** go: the names of deployed instances, private domains, a
  registry host, a personal email domain. `scripts/secret-scan.sh` appends the
  file to layer 1 when it exists.

**Layer 2 is optional.** A fresh clone, a new machine and a first run all work
without it: the scan runs, and passes, on layer 1 alone. What you lose is the
site-specific half - a deployed instance name pasted into a document is caught
on a machine that has the file and missed on a machine that does not. The
summary line says which layers ran, so a run is never ambiguous about it.

Write the file yourself, per machine. It holds `[[rules]]` blocks and nothing
else - no `title`, no `[extend]`, because the two files are concatenated and a
duplicated key makes gitleaks reject the merged config. Rejection is a hard
failure, not a silent skip. Use obviously fake values as a model, and put the
real ones only in your own copy:

```toml
[[rules]]
id = "site-instance-name"
description = "Names of the deployed instances at this site."
regex = '''\b(?:acme-prod|acme-test)\b'''

[[rules]]
id = "site-private-domain"
description = "This site's private domain and its personal email domain."
regex = '''[a-z0-9-]+\.(internal\.example\.com)\b'''
secretGroup = 1

[[rules.allowlists]]
description = "Files that must carry the pattern to define or test it."
paths = [
  '''(^|/)scripts/tests/secret-scan-gate\.test\.sh$''',
]
```

Two mechanics are worth knowing before writing a rule of your own, both
verified against the pinned binary rather than assumed. gitleaks' bundled
default configuration carries a **global allowlist** that silently drops any
finding whose secret is a whole absolute path under `/home`
(`^/(?:bin|etc|home|opt|tmp|usr|var)/[\w ./-]+$`) or begins with the letters
"true" (`(?i)^true|false|null$` - the `^` binds to the first alternative only).
A rule that reports its whole match therefore detects on some inputs and not
others, with no output either way; report a narrow `secretGroup` instead. And
`targetRules` does not exist in this version, so an allowlist is either global
or written inside the rule it applies to.

Two files at the repo root configure the scan:

- **`.gitleaks.toml`** extends (not replaces) gitleaks' default rule set.
  `[[allowlists]]` #1 covers generated/vendored directories - `target/`,
  `build/`, `.git/`, `.venv/` - that are never hand-authored and only slow the
  scan down. `.flatpak-builder/` and `.claude/worktrees/` are deliberately
  **not** on that list: the #811 incident's own leak involved a live `.env`
  *and* eight stale `.claude/worktrees` checkouts, so excluding either
  directory would reopen half the blind spot this gate exists to close - and
  because `.claude/worktrees/<session>/` nests full checkouts of this same
  repo, its tracked files (including this repo's own known-fake test
  fixtures) exist at N+1 different paths at once. The remaining
  `[[allowlists]]` cover this repo's own reviewed false positives (test-only
  PEM key fixtures, a truncated placeholder key inline in a unit test, a
  truncated example JWT in docs), matched by **repo-relative path suffix**
  (whole-file fixtures - the pattern names the tracked path, e.g.
  `crates/daemon/src/config/testdata/oidc_test_key\d+\.pem$`, never a bare
  filename and never a directory-wide exclusion) or by **content**
  (`regexTarget = "line"` or the extracted secret itself, for a single
  known-fake line inside an otherwise normal file, so nothing else in that
  file is exempted). Either way the match survives duplication: the same
  tracked fixture is exempt wherever `.claude/worktrees/` happens to have
  checked it out, not just at its canonical path. A path-**exact**
  `.gitleaksignore` fingerprint does not have this property - see below. The
  two private-information rules each carry their own allowlists, scoped to
  the rule rather than declared globally: the placeholder account names that
  are not people (`user`, `example`, `assistant`, and the synthetic names
  this repo's own fixtures use), and the two files that must contain these
  patterns in order to define and to test them - `.gitleaks.toml` and
  `scripts/tests/secret-scan-gate.test.sh`. Without that second exemption a
  clean checkout fails its own gate, which is the noise that trains people to
  bypass it. Scoping it to the rule keeps a real credential in either file
  caught by the default rule set.
- **`.gitleaksignore`** lists reviewed false positives by gitleaks fingerprint
  (`file:rule-id:start-line`) - currently empty. This mechanism is
  path-exact, which is right for a genuinely one-off finding but wrong for
  any of this repo's own tracked fixtures: a fingerprint for
  `crates/daemon/src/config/testdata/oidc_test_key1.pem` does not cover the
  byte-identical tracked file at
  `.claude/worktrees/<session>/crates/daemon/src/config/testdata/oidc_test_key1.pem`,
  so every stale nested worktree reproduced the whole fixture set as "new"
  findings - 27 false hard-failures scanning the primary checkout (3
  known fixtures x 9 stale worktrees) before this was caught in review.
  Reach for `.gitleaks.toml`'s path-suffix or content allowlists first;
  reach for this file only when a finding truly cannot be expressed that
  way.

Like the advisory scan, the step is deliberately unable to pass by accident:

- **gitleaks not installed** - hard failure, with the install command.
- **Installed version does not match the pin** - hard failure, naming both
  versions. Diagnosed separately from the next case, because they have
  different fixes.
- **Installed gitleaks reports no parseable version** (e.g. the CachyOS/Arch
  packaging defect above) - hard failure, with its own message: this is not
  version drift, so it is not worded as one.
- **A scan that produced no report** - hard failure. An exit status alone is
  not proof a scan ran (the same #706 failure mode `scripts/audit.sh` guards
  against): a fatal gitleaks error (bad config, bad path) exits non-zero and
  writes no report at all, so report-existence, not exit code alone, is what
  the script trusts.
- **A malformed host-local rules file** - hard failure, by the same
  report-existence check: gitleaks rejects the merged config and writes no
  report, so a syntax error in a hand-written layer 2 cannot read as a clean
  scan. A *missing* file is not a failure; it is the normal state on a
  machine that has not set one up.
- **A host-local rules file that exists but cannot be read** - hard failure,
  with its own message. Present-but-unreadable is not the same state as
  absent: the site-specific rules were meant to apply and did not, so the
  scan is missing a layer it was asked for and does not report clean.
- **A secret found** - hard failure, always. There is no suppress-and-warn
  path for a real finding the way there is for an informational RustSec
  advisory; add a reviewed `.gitleaksignore` entry for a genuine false
  positive, or rotate and remove a real one.

Either of the two version failures above can be overridden with
`ADELE_SECRET_SCAN_ALLOW_UNPINNED=1` (mirrors `ADELE_AUDIT_ALLOW_STALE`'s
shape and purpose in `scripts/audit.sh`): the scan still runs, against
whatever gitleaks is actually on `PATH`, and says loudly that the pin was not
verified. The exact pin stays the default - determinism in a security
scanner's rule set is worth keeping - but nobody should have to edit the gate
or reach for `--no-verify` to unblock work unrelated to the scanner itself.

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

### 8.3 The wire contract is the product (addition)

Adele is an API-first platform. The `Command` / `CommandResult` / `Event` surface in
`crates/api-model` is consumed directly by third-party clients, automations, and other
agents - the shipped clients are only its first consumers. Design every change to that
surface as a published contract, not as plumbing behind a user interface.

Base rule 8.3 already requires the business outcome in the payload. Three things follow from
it here:

**A refusal is a structured result, not a string.** A caller must be able to tell "you lack
permission" from "the database is down" without matching English text, so a rules-based
decline carries a stable machine-readable code, a description, a user-facing message, and a
`retryable` flag. Base rule 8.2 also applies: a decline is a normal outcome, so do not log it
at error level.

**Extend with optional fields, never with a new enum variant.** Serde fails to deserialize
an unknown variant, so a new arm on a wire enum breaks every client that has not been
rebuilt, including ones this repo does not ship. Add
`#[serde(default, skip_serializing_if = "Option::is_none")]` fields to an existing shape
instead. `WsHandshake` in `crates/api-model/src/lib.rs` documents this pattern where
`system_id` and `host_label` were added without changing the wire bytes for older clients.
Prove it with a test that parses the old shape.

**Serde compatibility is not source compatibility. Sweep the sibling repos.** An optional
field keeps old *payloads* parsing, and does nothing for old *code*: a downstream repo that
constructs the struct with an explicit field list fails to compile the moment the field
exists, and one that matches a struct variant without a `..` rest pattern fails the same way.
The clients in this group consume these crates by path or by a git dependency on `main`, so
there is no version to gate on - their build breaks as soon as this repo's `main` moves.

So a change to any shared type carries a sweep of `adele-gtk`, `adele-tui`, `adele-kde`,
`adele-web-ui`, `client-ui-common` and `voice` before it lands, and lands together with any
fix it needs. This is not confined to `api-model`: any type a client can name counts,
including configuration structs in the MCP and client crates.

Two traps make a hand search unreliable. Bare `grep` in the agent shell wraps `ugrep` with
`--ignore-files` and does not follow symlinked directories, and this group path-deps its
siblings through symlinks - use `command grep` or `rg`. And a construction site can hide
behind a type alias or a re-export, so search the field names as well as the type name.
Compilation is the only authoritative axis: build each consumer with `--all-targets`, which
includes test code, where most of these sites live. Enumerate consumers from the dependency
graph rather than from memory.

**Document the contract in the same change**, in `docs/API_TRANSPORT.md`,
`docs/WEBSOCKET_API.md` and `docs/dbus-api.md`. Where a change alters what an existing
caller sees - a command that starts refusing, a tool that starts declining, an environment a
subprocess stops inheriting - say so plainly as a behaviour change and say what an integrator
must do about it. The same applies to a tool schema advertised to the model: if an argument
is now ignored or constrained, the advertised description says so, because a schema that
promises what the code does not honour is a false contract.

### 9.1 Tracker for this project

GitHub Issues on `github.com/adelie-ai/desktop-assistant`, together with the shared `adelie-ai` project
board `Adelie AI Roadmap` (project number 1). Manage entries with the `gh` CLI
(`gh issue create`, `gh issue list`, `gh issue edit`, `gh pr create`). Put a new issue on the
board with `gh project item-add 1 --owner adelie-ai --url <issue-url>`, which lands it in
Todo. The board states are Todo, In Progress, and Done.

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

### Every change carries its user story (addition)

Before a change lands, answer two questions in the pull request: how does a person meet this
behaviour, and does any client need a user-interface change? A correct enforcement change
with no user story is a broken product.

Two shapes cause most of the damage.

**Late failure.** A panel assumes its command succeeds. Add a gate, and the user types a
value, presses save, and gets an error. The client must learn its capability up front and
render the affected section as visibly unavailable with a reason, which usually means the
daemon must expose that capability through the API. Decide that while you build the daemon
side, not after a client reports it.

**A silent cliff.** A capability disappears part-way through a turn and the user sees a
refusal with no cause and no way forward. Emit one legible signal on the existing status or
event channel, and make the refusal text say what to do next. One signal for the turn, not
one per refused call.

This is **Capability-based degradation** above, applied to authorization and policy rather
than to optional operating-system services: surface why something is off.

State what a single-user desktop user sees. For well-designed work the answer is nothing at
all - no new setting, no new concept, no new interface. When that is not the answer, the
design is usually wrong, not the requirement.

Do not build the client interface from a daemon ticket. Client work belongs to the client
repositories, so file an entry in each affected one naming the concrete API, the commands or
behaviour that changed, and the expected rendering. A vague note is not a handover. Where a
change accepts a usability cliff on purpose, record the accepted cost and what would remove
it, so the next reader knows it was a decision and not an oversight.
