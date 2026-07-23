# Skill execution and portability (design)

> Status: **mixed.** The catalog semantics described in "The catalog is
> cumulative" are **implemented** (#638, #639). Everything from "Portability"
> onward is a **design of record** — filed as #649, #650 and #651, none of it
> built. File and type references were verified against the code at writing time.

Companion to `skill-library-and-workflows.md`, which covers what a skill *is* and
how it is indexed. This one covers what happens when a skill has to actually run
somewhere: where its files live, whether a machine can execute them, and who
decides.

## The problem in one line

A skill is two different things wearing one name. Most skills are **prose** — a
procedure the model follows using tools it already has. Some skills also carry
**runtime scripts**, and those are only useful on a machine that holds the files
*and* has what they need to run.

Almost every hard question in this document comes from conflating the two. Kept
apart, the picture is simple:

| | prose skill | skill with runtime scripts |
|---|---|---|
| What it needs to be useful | the catalog row | files, a runtime, an executor |
| Portable across machines | already, once indexed | needs the bundle store (#649) |
| Runtime dependencies | none | the whole question (#650) |
| Execution gate | nothing to gate | the reason it exists (#576) |
| Containerization | never | the candidate (#651) |
| `present_on_disk = false` | cosmetic | usable to read, not to run |

The machinery below applies to a small subset of a small library. That is the
main argument for building it only when a skill needs it.

## The catalog is cumulative

*Implemented: #639.*

The database is the authoritative copy of a skill, not a shadow of the last scan.
Skills accrete; nothing is deleted because a scan stopped seeing it.

Before this, both reindex paths were replace-and-delete over their whole scope,
and the Postgres adapter said the quiet part in a comment: *"An empty scan clears
the whole global catalog."* A root that was briefly unreadable, a partial scan
that reached two roots of three, a home directory belonging to a client that
happened to be offline — each of those cost skills permanently.

What absence changes is not whether a skill exists but what still works. The body
reads fine; `disk_path` and attachments no longer resolve. `present_on_disk` and
`last_seen_at` record exactly that and are surfaced in the `builtin_skill_*`
payloads. Absent skills stay in search results deliberately — hiding them would
quietly reintroduce deletion by another name.

Removal became an explicit act (#640), and it needs almost no new machinery,
because catalog rows and per-user state are already separate: `skill_user_state`
is untouched by scans, so a rescan re-adds the catalog row while a user's
disabled state persists and the skill still never surfaces. Two cases follow —
files still on disk means per-user disable; files already gone means a real
delete is safe, since nothing will re-add it.

### Why the policy left the storage adapters

`reindex_global` / `reindex_for_owner` were *policy* verbs sitting on a storage
port, and the two implementations had already drifted: Postgres pruned by
name-list and preserved embeddings, SQLite deleted the scope wholesale.
Identical inputs, different catalogs, depending only on which store was
configured.

The port now exposes primitives — `upsert`, `list_scope`, `set_presence`,
`search`, `get`, `list` — and `core::skill_catalog::reconcile_scan` owns the
decision once. A trait pins signatures, not semantics, so the guarantee is
enforced by an executable contract: `core::ports::skill_index::conformance` runs
against the Postgres adapter, the SQLite adapter, and an in-memory reference
implementation. The last needs no database, so parity is checked on every test
run rather than only when someone has Postgres handy.

Dropping deletion also removed the need for a transaction. The reconcile used to
need one because a half-finished pass could delete; now the worst it leaves is a
stale presence flag that the next pass corrects, and the pass is idempotent.

One boundary remains open (#645): presence is reconciled per *scope*, but a scope
is scanned from several roots at once, and `main.rs` drops unreadable roots with
`is_dir()` before scanning. A root that is missing at boot therefore has its
skills flagged absent even though nothing looked at them. Presence belongs to the
root a skill came from.

## Skills are written for other harnesses

*Implemented: #638.*

The index deliberately aggregates other tools' libraries — `default_user_roots()`
scans `~/.agents/skills`, `~/.claude/skills`, `~/.codex/skills` and
`~/.cursor/skills-cursor`. Many of those skills are harness-agnostic. The ones
that teach how to do something *with the harness itself* name tools, commands and
UI that do not exist here.

The prompt used to say to read a skill body and "follow them as authoritative",
full stop. It now scopes that authority to intent rather than mechanics, and
directs translation over replay: keep the goal, the ordering and the domain
steps; substitute local tools; drop steps with no counterpart; say so and work
from first principles when a skill is really about another tool's internals.

## Portability: the bundle store

*Design of record: #649.*

A user-scoped skill's files live on whichever client authored them, which leaves
two gaps. Remote and k8s deployments have no story at all — the daemon cannot see
any home directory. And a skill is trapped on its author: write it on the desktop
and it is unusable from the phone, or after a reinstall, even though the catalog
remembers it perfectly.

So the client uploads a skill's files to the daemon, which can materialize them
back onto any client authenticated as the same user.

**Manifest and blobs, not an archive.** `skill_content_hash` already covers
`{relpath, sha256, mode}` per attachment. The client sends that manifest plus each
file's bytes, and the receiving side reconstructs the tree from paths it
validates itself. An archive parser in the trust path would bring zip-slip,
symlink and absolute-path entries, decompression bombs and duplicate-entry
shadowing; reconstructing from a validated manifest deletes that class outright
and reuses the hash the blessing already pins to.

**User-namespaced, RLS-enforced.** This is a different posture from
`skill_index` itself, which is deliberately host-global and outside RLS because
it holds metadata. Bundle bytes are user-authored content that later executes, so
they get the multi-tenant treatment: fail closed, never cross-tenant, and a
bundle ships only to a client authenticated as its owner. Blobs key on
`(user_id, sha256)` rather than bare `sha256` — a globally content-addressed
store dedupes across tenants and becomes an existence oracle. Global skills have
no bundles at all; they are baked into the image and already on the daemon's
disk. In k8s the store is PVC-backed.

**Materialize into a cache root**, never `~/.agents/skills`. That directory is
the user's authored space and codex/cursor read it too; a daemon-materialized
copy has to stay distinguishable from something the user wrote. The hash is
verified before anything is written.

**Disk wins for authoring; the daemon copy wins only for distribution.** A client
that has the files registers its own, and the daemon copy is materialized only
where they are missing. Nothing ever writes back over a working tree.

Only attachment-carrying skills get bundles. A prose skill is complete in the
catalog the moment it is indexed.

## Having the files is not being able to run them

*Design of record: #650.*

`scripts/run.py` needs python wherever it runs. Distribution alone just means the
same skill can now fail on four machines instead of one.

**Derive the requirement.** Skills from the shared agent directories declare
nothing, so declaration alone is useless. The scanner already reads every
attachment's bytes in order to hash them, so it can read the shebang for free:
`#!/usr/bin/env python3` needs python3, a `.sh` needs a POSIX shell. Frontmatter
can extend that — `SkillFrontmatter.metadata` already preserves unknown keys — but
absence of a declaration must never read as "no requirements".

**Report the capability.** Each runner publishes what it actually has:
interpreters and versions, in the shape of `builtin_sys_props` and
`adele-voice check-setup`, distinguishing "absent" from "present but the probe
failed" per the capability-degradation standard.

**Match at the gate, not at search time.** A skill whose scripts cannot run here
is still a good procedure to read, so discovery is unaffected; the execution gate
refuses, naming the requirement, this runner's gap, and a runner that satisfies
it.

Two limits are deliberate. Nothing may auto-install a runtime — an agent that
installs packages to satisfy a skill is a supply-chain event with extra steps.
And shebang detection finds interpreters, not libraries; a missing import fails
loudly and is surfaced verbatim rather than pretended away.

## Containerized execution as a third locality

*Design of record: #651.*

A skill with runtime scripts is built into an image, tagged in a user-namespaced
repository, and run — on the daemon's own container runtime, or as a job on a
cluster with a volume carrying the data to operate on.

This is a **third execution locality** beside `Daemon` and `Client` on the axis
that already exists. It replaces neither: opening an editor or inspecting a
laptop's session cannot happen in a pod. Containers are for compute.

Two things make it worth the weight. It dissolves the dependency question — the
skill carries its runtime, so capability matching reduces to *which localities
can run this*. And it supplies the isolation the execution gate has been
compensating for in software: `terminal-mcp` is `sh -c` with no sandbox, which is
why the gate has to be so careful and why an embedded interpreter was rejected on
privilege-cliff grounds. A container is a real boundary — no ambient daemon
credentials, no host filesystem, no cluster network unless granted.

### Commitments

**Content-addressed tags.** Tag the image with `content_hash`, so the blessed
hash and the image tag are the same fact: a different image is a different tag,
and an unapproved image cannot run by construction. Build caching falls out.

**Build outranks run.** Building from user content executes arbitrary build-time
code and produces a persistent, shared, cross-machine artifact. It needs its own
approval and a rootless builder — never a mounted container socket.

**Mechanics soft, authorization hard.** Orchestration belongs in an MCP server;
the daemon has no business learning cluster APIs. But the model calls MCP tools
directly, and a server that builds images and launches workloads would be the
most powerful tool in the fleet. That is exactly the shape of the `builtin_db_query`
self-blessing hole: a committing tool running as owner, reachable straight from
the model's tool list. So the daemon decides that this skill at this hash may be
built and run, and the server is reachable only through that decision.

**An orchestrator expressed as a skill must be global.** It cannot itself run in
a container — it bootstraps them, holding registry and cluster credentials — and
#605 lets Adele author skills, so an unconstrained version admits self-authored
content rewriting the orchestrator. Scope-follows-root already makes the rule
structural rather than a policy to remember: the orchestrator lives in an
operator-controlled global root, never user-scoped, never bundle-distributed,
never model-writable.

**Latency.** A full image build per invocation is absurd for a ten-line
procedure. Default to a base image per runtime with the skill mounted; bake only
when a skill declares real dependencies.

## The locality is derived, never chosen at runtime

This is the load-bearing principle, and it applies to all of the above.

Offering a mechanism whose use is a per-invocation judgment call puts the least
reliable component in the system at the exact point where the security posture
changes. A model choosing where a script runs is choosing a blast radius, and
model judgment under that kind of load has already failed here once, on record.

Three rules keep the decision out of the loop:

- **Derive it** from facts the system holds: does the skill carry executable
  attachments, what runtime does it need, which localities exist here. A rule,
  evaluated identically every time, testable as such. The model chooses which
  skill; it never chooses where the skill runs.
- **Pin it at approval time.** The user already approves a skill at a hash;
  approving the locality alongside makes it a stored, auditable fact, and a
  change of locality re-asks — which it must, being a different blast radius.
- **Collapse the option.** The strong form is not "containers are available for
  script-carrying skills" but *if a container runtime is present, script-carrying
  skills always run containerized*. No branch, no decision. Direct execution is
  the fallback where no container runtime exists, surfaced loudly because it is
  the weaker posture.

Stated generally, and worth applying past this document: **if you cannot derive
when a mechanism activates, it probably should not be optional.** Make it
always-on under a detected capability, or do not build it. Optionality is where
mid-tier model judgment leaks.

## Open questions

- The data contract for containerized runs: per-run ephemeral, per-skill
  persistent, or caller-specified.
- Registry location, credentials, and garbage collection — hash tags churn on
  every skill edit.
- How a pod's exit status and output become a tool result the model can reason
  about.
- Whether approval is per `(skill hash, locality)` or per skill hash with the
  locality disclosed and re-asked on change (#575). The second fits the
  ask-once-and-remember flow, but only if the locality is genuinely shown at
  approval time alongside the daemon-extracted facts.
