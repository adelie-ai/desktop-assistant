# Skill library and workflows (design)

> Status: **proposed, not yet implemented.** This is a design of record, not a
> description of shipped behavior. File and type references point at the code the
> feature would build on, verified at design time.

## What this adds, and why

Two capabilities, layered:

1. **A skill library.** Skills already live on disk as `SKILL.md` playbooks under
   several agent directories (`~/.agents/skills`, `~/.claude/skills`, and the
   Codex/Cursor equivalents), and `skills-mcp` can already create, read, and
   substring-search them. This layer makes the daemon **index those skills into a
   vector-searchable catalog linked back to disk**, so the assistant can find the
   right playbook by meaning, review it, and use it on its own -- without the user
   naming it.
2. **Workflows.** A workflow is a skill whose body carries a numbered `## Steps`
   list. `run_workflow` expands it deterministically into the conversation plan,
   and the assistant executes it under the existing `[Plan]` machinery with
   mechanical checks that the steps are actually followed in order. The value is
   **conformance** -- the same procedure, run the same way each time -- not saving
   tokens.

Neither layer introduces a scripting engine. The assistant does not write and run
Lua/Python/JS to "execute" a workflow; the steps are prose the model follows under
enforced ordering. A structured step-runner is a possible future (folded into the
durable scheduler, da#552), not part of this design.

### Why not an embedded interpreter

Recorded so the question is settled rather than re-litigated. Embedding Rhai/Lua/
QuickJS/WASM, or having the assistant author a Python script run via `terminal-mcp`,
was evaluated and rejected for this system:

- **Authorability.** A mid-tier local model authors these skills. A numbered
  markdown checklist is the artifact such models produce and follow most reliably;
  free-form code is the artifact they confabulate (a documented failure on this
  repo). A save-time compile gate validates syntax -- the thing they rarely get
  wrong -- and cannot validate result-shape assumptions -- the thing they do.
- **Determinism of value.** The stated value is conformance, not tokenless
  mechanical execution. An interpreter buys the latter, which is not wanted.
- **Security.** `terminal-mcp` is `sh -c` with no real sandbox; an assistant-authored
  script is a privilege cliff that defeats tool allowlisting. `mlua` is unsafe C FFI
  in-process.
- **Durability.** Replay-based script resume consumes journal entries positionally
  and silently mis-executes on any drift; this daemon restarts primarily via batch
  redeploys, making drift the common case, not the rare one.

## Layer 1: the skill library

### Scope follows location

A skill's **scope is inherited from the root directory it lives in**. No per-skill
scope metadata is needed.

- **Global roots** -> global skills, available to every user on the host. A
  **configurable list** with a platform-appropriate default:
  - Linux package/distro: `/usr/share/adelie/skills`
  - macOS / Homebrew: `<brew-prefix>/share/adelie/skills`
  - Container: a baked-in path (see below)
- **User roots** -> user-scoped skills, visible only to their owner:
  `~/.agents/skills`, `~/.claude/skills`, `~/.codex/skills`, `~/.cursor/skills-cursor`.

Because roots are declared global-or-user in config, a same-named skill in a user
root **shadows** the global one for that user (the `~/.local/share`-over-`/usr/share`
pattern).

### Locality follows scope (the client-vs-daemon axis)

Skills reuse the locality model tools already have (the client-side MCP host,
epic #464, and runner labeling, da#531):

- **Global skills are daemon-side.** The daemon scans its configured global roots at
  startup.
- **User-scoped skills are client-side.** The client scans the user's home roots and
  registers them with the daemon via a new `RegisterClientSkills` command -- the
  mirror of the existing `RegisterClientTools` (`crates/api-model/src/lib.rs`). In a
  containerized multi-user split the user's skills ride in from the client while the
  baked-in skills stay global; in a co-located local install it is the same path,
  the client simply happens to be local. (A daemon reading the shared home directly
  when co-located is an optional optimization, not a second architecture.)

The payoff: **attachment-script execution lands on the correct runner for free.** A
user-scoped skill's bundled scripts live on and run on the client (client-run
`terminal`/`fileio`); a global skill's scripts run in the daemon. The execution gate
(see Safety) is enforced at whichever runner hosts the skill.

### Container deployment

Global skills are baked into the daemon image, matching the MCP-fleet image pattern
(`FROM <base>` + `COPY`). Document the baked path (a configured global root) so an
operator can:

```dockerfile
FROM adelie-daemon:base
COPY my-skills/ /usr/share/adelie/skills/
```

(Path prefix -- `/usr/share/adelie` vs the fleet's `/opt/adele/mcp` -- should be
reconciled to one convention across the project.) User-scoped skills in a container
arrive via client registration; a per-user volume is a later addition.

### Storage: catalog vs per-user state

Two tables, deliberately separated so the disk sync never touches user decisions.

**`skill_index`** -- the catalog. Disk-synced; **host-global, modeled on
`tool_definitions`** (no forced per-user scope), with a **nullable `owner_user_id`**
(`NULL` = global, set = user-scoped). Columns: name, `disk_path` (the backlink),
frontmatter, body text, derived `kind` (`skill` | `workflow`), attachment manifest,
`content_hash`, source/`trust_tier`, `locality` (daemon | client), embedding,
`indexed_at`.

- *Why not the knowledge base:* KB consolidation loads all entries with no source
  filter and is told to prune/merge/delete them
  (`crates/storage/src/dreaming/consolidation.rs`). Disk-canonical skills placed in
  the KB would be silently rewritten by the dream cycle. A separate store keeps disk
  authoritative.
- *Why host-global:* the startup scanner has no request scope, so
  `current_user_id()` collapses to the `"default"` sentinel
  (`crates/core/src/ports/auth.rs`). `tool_definitions` is the correct precedent (no
  `user_id`, not in the RLS backstop). User-scoped rows set `owner_user_id`
  explicitly and are populated per-user (client registration, or a
  loop-over-known-users scan like the dream cycle re-installing scope).

**`skill_user_state`** -- per-user, RLS-scoped: `(user_id, skill_ref, enabled,
approved_for_use, approved_for_unattended, blessed_hash, blessed_at)`. This holds
**both** enablement and blessing. Because it is keyed by user and never part of the
disk scan, the scanner can freely sweep and rewrite the catalog without disturbing
anyone's enable/disable/approve decisions -- which also removes any need to retain a
catalog row just to remember it was disabled.

### Indexing lifecycle

Model on the tool-definition startup path (`crates/daemon/src/main.rs`,
`provider_reindex`):

- **At startup**, scan the global roots and upsert rows with **NULL embeddings**.
  Rows are immediately FTS-searchable; vectors are filled by a background
  `backfill_skill_embeddings` pass reusing `EmbedFn` + `EMBED_TIMEOUT`. Boot never
  blocks on a cold embedding backend, and a down backend degrades to FTS-only.
- **Freshness** beyond boot: a `resync_skills` control (modeled on the runtime
  `reindex_source` seam, which atomically sweeps and rewrites one source), plus a
  single-path reconcile when the assistant authors a skill through `skills-mcp`
  (otherwise a skill it just created is invisible until restart).
- **Search** reuses the hybrid vector+FTS (RRF) *pattern* from the KB / tool
  registry, copied into a `skill_index` adapter (the RRF SQL is table-bound; this is
  copy-the-shape, not call-into-KB). Two tools: `builtin_skill_search {query, kind?,
  limit?}` and `builtin_skill_get {name}` (returns body, resolved `disk_path`,
  attachment paths, trust tier, blessed state).

When Layer 1 is enabled, `skills-mcp`'s own `search`/`list` verbs are demoted --
`skills-mcp` remains the **authoring/write** surface only -- so the assistant sees
one discovery tool, not several near-synonymous ones.

### Portability and degradation

Where no global root resolves and no client registers skills (e.g. a remote-brain
split with no shared filesystem and no client scan), the feature is **cleanly off**,
logged once at startup, with `skill_search`/`skill_get` returning an explicit
disabled result rather than an empty mystery catalog. Local/systemd is the primary
target for the first cut.

### Shared-directory discipline

The agent directories are shared across products (Codex, Cursor, Claude, the vercel
`find-skills` installer) with **pre-defined formats**. Therefore:

- Shared roots are indexed **read-only**. We never rewrite their `SKILL.md` files,
  never add our own frontmatter fields, never write their `.skill-lock.json`.
- The `workflow` kind is derived from a `## Steps` section in the **free-form body**,
  not a non-standard frontmatter tag, so even our own workflows stay
  ecosystem-compatible.
- The assistant authors its own skills/workflows into an **Adele-owned write root**,
  not into the shared tree.
- All state we own (blessing, enablement, kind, index metadata) lives in **our
  database**, never as sidecar files in shared directories.

## Layer 2: autonomous discovery and blessing

The intended flow, in the user's words: *the assistant looks for a skill, finds one,
asks if it is OK, and remembers the answer.*

- When a request looks procedural or matches a known playbook, the assistant
  searches the library, reviews the top hit for fitness, and either uses it or -- if
  there is no blessing on record for this user -- asks for approval with a summary.
- **Remembering the answer** is the per-user state: **yes** records
  `approved_for_use` (blessed; never asks again); **no** sets `enabled = false` for
  that user (won't nag). Pre-blessing is the same approval issued ahead of use.

### The non-forgeable blessing seam

A blessing must not be forgeable by the model. The real boundary is the **channel**,
not identity: `user_id` is per-connection and wraps the model's own turn identically
(`crates/daemon/src/transports.rs`), so it cannot separate user from model. But the
model can only emit LLM tool calls; it has **no path to emit a client->daemon
`Command`** (Commands are decoded transport-side only,
`crates/transport-dispatch/src/lib.rs`). Therefore:

- A blessing is written **only** by the handler for a new **`Command::BlessSkill`**
  on the peer-cred-authenticated channel (DA #407). The model can *propose* and
  *summarize*; it cannot record its own approval.
- The daemon **re-derives** the skill identity and recomputes the content hash from
  disk when handling `BlessSkill`; it never trusts a model-supplied hash. The
  confirm affordance should be a client action bound to the exact skill and hash the
  user is viewing, not free-text "yes" detection.
- The approval surface renders **daemon-extracted facts** -- the raw `## Steps`
  verbatim, the literal attachment filenames and which are executable, the exact
  tool names the daemon parsed, and the source/trust tier -- because a prompt-injected
  body can shape a benign-sounding model summary. Model prose may accompany these
  facts, never replace them.

### Trust tiers

Carry `source`/`sourceType` from `.skill-lock.json` into the catalog as a trust tier
(`local` | `github` | `well-known` | `unknown`). Remote/third-party skills get the
stronger disclosure, can never be `approved_for_unattended` on first bless, and
re-ask on any source change.

## Layer 3: workflows

A workflow skill body:

```markdown
## Parameters
- month: billing month, e.g. 2026-07
- dry_run = true: skip mutating actions

## Steps
1. Preview invoices for {{month}} and summarize totals.
2. (approval) Finalize all invoices for {{month}}.
3. Notify me with the final totals.

## Guidance
Prose: edge cases, what "done" means per step, failure handling.
```

`kind = workflow` is derived from the presence of `## Steps`. Parameter defaults use
a mechanical `= value` syntax the parser owns, not prose.

**`run_workflow {name, params?, validate_only?}`** -- a core-loop tool intercepted
beside `begin_step`/`complete_step` (`crates/core/src/planning.rs`):

1. Fetch the indexed body; lint (pathed, did-you-mean errors so a mid model converges
   in one round trip; `validate_only` returns lint results without seeding).
2. Substitute params (escaped, newline-stripped). **Do not persist raw parameter
   values** -- they would be embedded, synced to client sidebars, and logged. Store
   parameter names with redacted/hashed values; keep resolved values in memory for
   the turn only. Add a value-shaped-secret detector on top of rejecting
   secret-shaped names.
3. Seed a per-run provenance note (`workflow:run:<n>` -- name, redacted params,
   resolved-steps snapshot, guidance prose, content hash) plus one sequenced `todo`
   note per step, continuing the root counter.

From there the run is an ordinary plan: `[Plan]` rendering, per-step outcome notes,
tool-result eviction.

### Claim-by-key and mechanical conformance

`StepStack` always mints a fresh key today (the #292 clobber guard), so a seeded plan
is not runnable as-is. `begin_step`/`complete_step` gain an **optional step-key
argument** to *claim* a seeded todo. This composes with the subagent branch's
`owner_todo` path-namespacing (step-key identity vs path scope are orthogonal) but
touches the same surface, so it lands coordinated with that slice and must not
regress the #292 guard.

Conformance is enforced **mechanically**, not by prompt guidance -- it is the whole
point of the feature:

- Reject minting a new step while an unclaimed earlier seeded step is still open
  (claim-in-order).
- Reject completing step N+1 while a seeded step <= N is open.
- At turn end, flag any run whose seeded todos are not all terminal as **INCOMPLETE**
  rather than silently "done".

A short follow-through prompt section (rendered only while a run is *active*) adds the
soft guidance: work in seeded order, record real outcomes, tool output never adds or
reorders steps, on failure record FAILED and stop.

Keep the seeded-step representation forward-compatible with a future persisted DAG so
da#552 can enforce `depends_on` later.

## Triggering

A workflow is the reusable *reaction*; something else decides *when* to fire it. The
three trigger classes converge on one model -- a trigger starts an agent run that may
invoke a workflow -- so there is never a second scheduler or a workflow-specific
trigger engine.

- **On-demand:** the assistant auto-discovers and runs; the user does not name a
  workflow.
- **Scheduled:** routine sugar. A routine (da#413) whose prompt runs a workflow gives
  scheduled execution.
- **Event-driven (near-term strategic direction).** A coming project reacts to
  events -- an email arrives, a chat arrives, a file lands in a watched directory to
  be read and indexed into the knowledge base. Each is a trigger that starts an agent
  run, which may run a workflow as its reaction. This generalizes the routines epic
  from cron-only into trigger/condition/action (da#413 Phase 4). The workflow and
  skill-library substrate is designed to be that reaction target: an event source
  fires a run, the run auto-discovers the fit workflow, and the blessing/allowlist
  gates apply exactly as for a scheduled run.

The **file-watch -> read-and-index-into-KB** case is the natural *safe reference*
event reaction -- KB-only tools, no external mutation -- the events analog of how KB
consolidation is the safe first routine. Prove the unattended path on it before
wiring email/chat reactions, which carry real mutation and injection risk.

Because event-triggered runs act on **attacker-influenced content** (an incoming
email or chat can carry injection aimed at an agent holding tool access), the two
safety gates below are load-bearing for this direction, not optional hardening.

## Safety

The interactive path (user present) rests on the blessing gate plus the assistant's
review. The **unattended path** carries two hard gates that must be closed before any
scheduled mutating run ships, independent of the routine runner:

1. **The content hash must cover attachment bytes.** Attachments are executable
   scripts. If the hash covers only `SKILL.md` (or reuses `.skill-lock.json`'s
   `skillFolderHash`, which is empty for well-known sources and computed by an
   external tool), a script swap under a blessed, byte-identical `SKILL.md` keeps the
   blessing valid -- a persistent scheduled RCE. Adele computes its own
   `sha256(SKILL.md + sorted manifest of {relative_path, sha256, mode})`. Any
   attachment change invalidates the blessing. (`sha2` must be promoted to a normal
   dependency; today it is optional/PKCE-only.)
2. **Script execution needs a mechanical gate.** The standing system prompt tells the
   model to use `terminal`/`fileio` without asking, so a prompt-only bless is bypassed
   by simply running the script. The daemon (or the hosting client, per locality)
   refuses to exec/read a path under a scanned skill root unless that skill's current
   hash is blessed for this user -- or routes attachment execution through a single
   bless-checking builtin. Skill-attachment execution is carved out of the
   "don't ask permission" default.

Additional posture: destructive tools stay off unattended allowlists by default; an
optional owner-maintained destructive-tool list (config, not model-writable) forces
an approval marker; `builtin_db_query` is already SELECT-only (not a write-forge
vector) but stays out of workflow-running contexts for hygiene; the anti-forge
guarantees are that blessing is writable only via `Command::BlessSkill`, is kept out
of the model-writable scratchpad, and is pinned to the attachment-covering hash.

## Phasing

External gates named explicitly: **da#544 inc2** (sqlite-vec; limits only sqlite
semantic search, worked around by FTS fallback) and **da#413 Phase 1** (the routine
runner; blocks only the unattended slice).

1. **Skill index.** `SkillIndexStore` port + Postgres adapter (host-global, modeled
   on `tool_definitions`) + sqlite relational+FTS adapter (inc1 pattern); startup
   scanner over configurable global roots; the attachment-covering `sha2` hash
   (promote `sha2` here -- later slices pin to it); NULL-embed + backfill;
   Postgres semantic search now, sqlite FTS-only pending da#544 inc2;
   `builtin_skill_search`/`skill_get`; demote `skills-mcp` search/list. Ships
   immediate value: every existing skill becomes semantically discoverable.
2. **Discovery + blessing.** Prompt guidance for search/review/ask; `skill_user_state`
   table; `Command::BlessSkill` on the peer-cred channel with daemon-re-derived hash
   and a specific-skill confirm affordance in at least one client; trust-tier column;
   remember-yes-and-no. (Not daemon-only -- requires client work.)
   - **2b. Execution gate.** The mechanical attachment-execution gate. Precedes
     telling the assistant to auto-discover-and-use.
3. **Workflows.** Body convention + `workflow-authoring` meta-skill + `run_workflow` +
   claim-by-key + mechanical conformance + active-run follow-through. Coordinate
   claim-by-key with the subagent branch's scratchpad slice.
4. **Extra rails (deferred).** Reserved keys, destructive-tool list -- built once a
   real workflow proves one is needed. (The disabled-state is earned and lives in
   slice 1.)
5. **Unattended (last).** Routine sugar; `approved_for_unattended` + hash gate;
   read-only default allowlist; park-and-notify at approval gates; the fuller
   client-UI bless/approve action. Hard-gated on da#413 Phase 1 **and** both safety
   gates above.

## Open decisions

- **Default state of a global skill for a user.** Recommended: available and
  enabled-but-unblessed (surfaces, asks once, remembers), with per-user disable to
  suppress. The alternative is opt-in per user.
- **Unattended execution is confirmed in scope** (resolved). It is the on-ramp to the
  event-driven direction above, so slice 5 -- its client-UI approval surface and both
  safety gates -- is required, not optional. The remaining question is only
  sequencing against the events project (below).
- **Where the unattended/events seam lives.** Slice 5 hands the unattended-run gate
  (blessing + hash check, park-and-notify) to whoever builds the routine/event
  runner (da#413). Decide whether the event-source work is a sibling epic that
  consumes this substrate, or folded into da#413's later phases.
- **Full indexing of user-scoped client skills** (client ships bodies to the daemon
  for embedding -- user content crosses to the daemon in the container case) vs a
  lighter names-only registration with on-demand fetch. Recommended: full indexing,
  since uniform search over global and user skills is the point.
- **A local-model pilot** measuring claim-vs-mint and fitness-judgment error rates on
  the actual target model before relying on conformance/summaries unattended.
- **Path convention** reconciled across the project (`/usr/share/adelie` vs
  `/opt/adele`, "adele" vs "adelie").

## Key integration points

| Concern | Where |
| ------- | ----- |
| Startup index + reindex seam | `crates/daemon/src/main.rs` (`provider_reindex`, `reindex_source`) |
| Embedding + hybrid search pattern | `crates/storage/src/knowledge.rs`, `crates/storage/src/tool_registry.rs`, `crates/storage/src/embedding_backfill.rs`, `crates/core/src/ports/embedding.rs` |
| Plan / step machinery | `crates/core/src/planning.rs` (`begin_step`/`complete_step`, `StepStack`) |
| Tool dispatch | `crates/core/src/ports/tools.rs`, `crates/mcp-client/src/executor.rs`, `crates/mcp-client/src/builtin.rs` |
| Client registration mirror | `crates/api-model/src/lib.rs` (`RegisterClientTools`), `crates/transport-dispatch/src/lib.rs` |
| Peer-cred channel | DA #407, `crates/peer-cred`, `crates/daemon/src/transports.rs` |
| Background/unattended host | `crates/application/src/background_tasks.rs`, routines epic da#413 |
| Skills on disk | `skills-mcp` (`src/repo.rs`), `~/.agents/.skill-lock.json` |
