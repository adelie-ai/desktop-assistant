# Adele — project instructions

@AGENTS.md

Adele is a personal AI assistant: the `desktop-assistant` daemon plus its clients
(GTK, TUI, KDE, voice, web). These are the guiding principles behind how Adele should
*behave* and how we *build* it. Architecture is in `docs/architecture.md`; day-to-day
commands and conventions in `docs/development.md`.

## How Adele should behave

**Work like a human mind.** People summarize, re-contextualize, and strategically
forget — all the time. We don't hold every raw detail in working memory; we write
things down where we'll find them again. Adele should do the same:

- **Two memories.** The **scratchpad** is short-term working memory for the *current
  task* (plans, findings, status, decisions). The **knowledge base** is long-term
  memory for what's worth keeping across conversations. Write to whichever fits —
  don't keep important state only in the live context.
- **Summarize, don't hoard.** Once transient detail (a raw tool result, an
  intermediate step) is no longer needed, compact it to a summary and drop the bulk.
  Dragging along huge, expensive contexts full of irrelevant detail is a bug, not
  safety — a summary in the scratchpad or KB is almost always better.
- **Never lose data or important context.** Losing the user's prompt, a decision, or
  a key fact causes *wrong* behavior — that is the line summarizing must never cross.
  Strategically forgetting transient detail is good; forgetting what the work was
  *for* is not.
- **Tag at the right generality.** A saved fact should be general enough to be useful
  again later, but specific enough that it doesn't surface where it's irrelevant. Tag
  it so it's found when it matters and stays quiet when it doesn't.

**Recover gracefully, stay friendly.** When something goes wrong — a limit hit, a tool
failure, a backend error — recover naturally and keep going, almost as if there were
no problem at all. Apologize plainly if it helps, say what happened and how to
continue, and never dump raw errors or dead-end the user. A turn should never end in a
mess or vanish silently. (Example: exhausting the tool-round budget winds down with a
fluent closing and persists the turn, rather than erroring and dropping it — #453.)

**Degrade, never hard-depend.** Every optional OS/desktop integration (logind, screen
lock, KDE, PipeWire, any D-Bus service, audio devices) must be capability-detected and
degrade cleanly when absent — Adele may run headless. Distinguish "is this capability
present?" from "did this call succeed?", and surface *why* something is off rather than
failing opaquely.

**Multi-tenant, fail-closed.** User data is isolated by `user_id`. Prefer defense in
depth (e.g. the `db_query` tool is both AST-scoped *and* Postgres-RLS-enforced), and
when in doubt fail closed — return nothing rather than another user's rows.

**Native and fast, not a browser in a costume.** Desktop clients should be lean and
native (GTK, TUI, KDE), not Electron-style browser wrappers. A genuine web client for
the web/mobile/remote case is fine (`adele-web-ui`) — favor lean wasm over embedded
Chromium.

## How we build it

- **Spec-driven / TDD.** Write the failing test first; make each acceptance criterion
  a named test; enumerate the unhappy paths (empty, max, concurrent, cross-tenant,
  partial, malformed). Pin a bug with a test before fixing it.
- **Warnings are failures.** No compiler or clippy warnings, no ignored tests, no
  `--no-verify`. Fix the root cause, or suppress narrowly with a written reason.
- **Security review before every PR.** Adversarially read the diff and scan new
  dependencies for CVEs; high/critical findings are hard blockers, not advisories.
- **Small, low-cognitive-load changes.** Phase big work into small issues; avoid
  speculative abstractions; reuse existing traits; justify every new type.
- **Don't break `main`.** Merge each green, independently-shippable PR; land
  co-dependent PRs together and note the interlock.
- **Keep `crates/core` adapter-independent**, behind trait boundaries — see
  `docs/development.md` for the full conventions.
