# Skill library (on-disk skills, indexed and searchable)

The daemon indexes on-disk `SKILL.md` playbooks into a searchable catalog so the
assistant can find and read a reusable how-to by meaning. This page documents the
runtime behavior so it need not be re-derived from the code. The full feature
design (workflows, per-user blessing, client-registered user skills) lives in
`docs/design/skill-library-and-workflows.md`; this page covers what ships in the
Postgres index slice (#573).

> Scope: the **Postgres** path. The SQLite adapter (#594) and the skills-mcp
> search demotion (#595) are follow-ups; user-scoped (client-registered) skills
> are a later slice. Everything here is the host-global path.

## What a skill is

A directory `<root>/<name>/SKILL.md` — YAML frontmatter (`name`, `description`,
optional `tags`, plus any other keys preserved as `metadata`) followed by a
markdown body. Sibling files (scripts, references) travel with it as
**attachments**. A skill whose body has a `## Steps` section is a **workflow**;
otherwise a plain **skill**. The format is the shared cross-product one
(`~/.agents/skills` etc.), so the daemon reads it without inventing new fields.

## Where skills come from

`[skills]` in `daemon.toml`:

```toml
[skills]
enabled = true                      # default
roots = ["/usr/share/adelie/skills"]  # platform default; configurable
```

Global roots are **host-global** (owner-less), scanned by the daemon at startup.
The default is platform-appropriate (Linux `/usr/share/adelie/skills`; macOS the
Homebrew-prefix `share/adelie/skills`; a container bakes into whatever root it
configures). The list is configurable so a packager points at the right place
without a code change. When no configured root resolves, the feature degrades off
and logs once — it never blocks startup.

## How indexing works

`crates/daemon/src/skill_scanner.rs` walks each root, parses every `SKILL.md` with
the pure `core::domain::skill` helpers, and produces one `IndexedSkill` per skill:

- **Content hash** covers `SKILL.md` **and every attachment's bytes, path, and
  mode** (`skill_content_hash`), so a swapped script changes the hash — the
  integrity anchor a future blessing pins to.
- **Kind** is derived from a `## Steps` section; **trust tier** from a
  `.skill-lock.json` `sourceType` (github / well-known / local / unknown).
- Malformed or unreadable skills are skipped with a warning; earlier roots win a
  name collision.

The scan is handed to `core::skill_catalog::reconcile_scan`, which upserts every
skill it saw and marks the rest of that scope absent. A skill's embedding is
**preserved across a rescan unless its content hash changed** (so a boot rescan
doesn't re-embed everything). Rows land with a NULL embedding; the existing
embedding backfill loop fills them (`backfill_skill_embeddings`), degrading to
full-text-only when the embedding backend is down.

Storage is a host-global `skill_index` table (migration `031_skill_index.sql`) —
no `user_id`/RLS, modeled on `tool_definitions` — with hybrid vector + `tsv`
full-text (RRF) search.

## The catalog is cumulative

The database is the **authoritative copy** of a skill, not a shadow of the last
scan. Skills accrete, so Adele gets better over time, and nothing is ever deleted
because a scan stopped seeing it. That matters most for the cases where a scan is
simply *unable* to see something: a root that is momentarily unreadable, a home
directory belonging to a client that happens to be offline, a partial scan that
reached two of three roots. None of those may cost you a skill.

What absence does change is what still works. A skill whose files are gone still
reads — the procedure is intact and searchable — but its `disk_path` and
attachments no longer resolve, so its bundled scripts can't run. Two columns
record exactly that: `present_on_disk` (were the files reachable at the last scan
of this scope?) and `last_seen_at` (when that was). Both are surfaced in the
`builtin_skill_search` / `builtin_skill_get` payloads, so the model can tell the
difference between a procedure it can follow and one whose tooling has vanished.
Absent skills stay in search results deliberately: hiding them would quietly
recreate the deletion behavior this design removes.

Removal is therefore an explicit act, never inferred from a scan (#640).

**Reconcile policy lives in `core`, storage keeps only primitives.** The port
(`SkillIndexStore`) exposes `upsert`, `list_scope`, `set_presence`, `search`,
`get` and `list`; deciding what accretes and what is marked absent happens once,
in `reconcile_scan`. That split is not cosmetic. When each adapter implemented a
`reindex_*` verb of its own, the two drifted — Postgres pruned by name-list,
SQLite deleted the scope wholesale — and identical inputs produced different
catalogs depending only on which store was configured. A trait pins signatures,
not semantics, so the guarantee is enforced by an executable contract
(`core::ports::skill_index::conformance`) that the Postgres adapter, the SQLite
adapter, and an in-memory reference implementation each run as their own tests.

One consequence worth naming: the reconcile no longer needs a transaction. It
used to, because a half-finished pass could delete skills. Now the worst a
partial pass leaves behind is a stale presence flag that the next scan corrects,
and re-running a scan changes nothing after the first.

## Approval: whether a skill may be followed

`trust_tier` records **provenance** - where a skill came from (`local`,
`github`, `well_known`, `unknown`). It says nothing about whether anybody agreed
to run it. Approval is a second, orthogonal column pair (#1155):

| Column | Meaning |
| ------ | ------- |
| `approved_at` | When a person approved the skill. `NULL` means not approved. |
| `approved_by` | Who approved it. `NULL` on a single-person deployment. |

The two axes are genuinely independent. A skill fetched from GitHub can be
approved. A skill Adele wrote for herself is `local` - the most trusted
provenance the catalog has, because it really was authored locally - and must
still not be followed until somebody says so. One column cannot hold both facts.

Who may write the columns is deliberately narrow:

- **A scan approves what it inserts.** Putting a file in a skill root is a
  deliberate human act, so `reconcile_scan` stamps `approved_at` on every skill
  it inserts. That is the only place the scan path may decide it.
- **A rescan never re-approves.** `upsert` honours approval on insert and
  preserves it on update, so a skill a person unapproved stays unapproved
  through every later scan.
- **`write_authored` always lands unapproved.** It forces `approved_at` to NULL
  on both branches, so an amend of an approved skill drops the approval the old
  body earned. Forcing it in the store rather than in the caller is what makes
  that atomic: there is no window in which new content wears an old approval.
- **`set_approval` is the explicit flip**, in either direction. Nothing calls it
  yet, so a self-authored skill stays unapproved until an approve surface exists.

### What "not approved" actually withholds

The column would be decoration if nothing read it, so the read path enforces it:

- `builtin_skill_search` omits unapproved skills, and reports how many it held
  back. Filtered before the result limit, so an unapproved row can never
  displace an approved one from the results.
- `builtin_skill_get` refuses an unapproved skill by name, returning
  `ok: false`, `awaiting_approval: true`, and **no body**. The body is what gets
  followed. The refusal names the reason rather than pretending the skill does
  not exist, so the model does not simply ask again.
- The `[Recall]` block's skill arm never offers one (#1154). It cannot be
  marked and shown instead: `builtin_skill_get` refuses it, so the line is one
  the model can only fail on, and it would accrue an offer every turn it ranked
  near a prompt and never an open - the profile ranking reads as evidence to
  retire a skill. See `docs/features/pre-prompt-recall.md`.

Both descriptions advertised to the model say so, because a schema that promises
what the code does not honour is a false contract.

## Skills Adele writes for herself

When a plan finishes, the procedure it followed is already written down: the
scratchpad holds one `todo` note per step and one `outcome:<step>` note per
finding (#240), which is a `## Steps` workflow in everything but name. So a
finished plan is **promoted**, not authored from scratch.

**The trigger.** A plan comes back to the root, and the offer arrives inside the
`complete_step` acknowledgement as a `skill_offer` field. At most one offer per
turn: a turn may return to the root several times, and repeating the offer would
train the model to ignore it.

**The bar**, in `crates/core/src/skill_promotion.rs`, and what it excludes:

| Rule | What it keeps out |
| ---- | ----------------- |
| The plan read from the scratchpad did not hit its page cap | A plan that may be missing its later steps. The store returns `note`-typed rows before `todo`-typed ones, so a full page cuts the END of the plan, and a skill that stops halfway is worse than no skill. |
| At least 3 steps that finished **and** recorded an outcome | A question answered (no plan at all), a single file written (one step), a pair of acts with no shape between them (two steps), and any step whose finding was never written down |
| **This turn** did not read a skill (`builtin_skill_get`) before it planned | Re-saving a skill the turn just followed, which is how a library fills with near-duplicates. Searching the library is not following one, and a lookup in an earlier turn is not this plan following a skill |
| No more than a third of the plan abandoned | A plan that records a search rather than a method |
| The turn did not ingest external content (#741) | A turn whose own wording is withheld, and which therefore recorded no procedure |

**It is an offer, never a write.** The model has the context to say whether what
it just did generalises, and that judgement is the whole value. Declining is
doing nothing.

**Dedup happens before the offer.** The catalog is searched with the plan's own
goals, and any match is named in the offer. At the write, a request to add a
skill whose name is already taken is refused outright - never satisfied with a
second row. A lookup the catalog cannot answer refuses too: the write upserts on
`(name, owner)`, so reading a failed lookup as "the name is free" would replace
an existing skill and drop its approval.

**Amending is limited to the assistant's own unadopted drafts** - not approved,
not on disk, and written by the promotion or extraction path. Amending swaps the
body, relabels the provenance as self-authored, marks the row absent from disk
and drops the approval, so aiming it at a skill a person placed, approved, or
installed would destroy their work. The offer's `mode_hint` follows the same
rule, so the model is never steered into a refusal.

**Accepting** calls `promote_plan_to_skill {name, description, mode?, tags?,
summary?}`. The body is rendered from the plan's steps and outcomes; the model
supplies only how the skill is found and what it is for. The transcript is never
read, because the transcript carries the dead ends and the plan carries what
worked. The bar is re-checked at the write, so a plan that never cleared it
cannot be kept by calling the tool directly.

**What lands** is a catalog row scoped to the caller (never host-global), with
`trust_tier = local`, `source = self-authored`, `approved_at = NULL`, and
`present_on_disk = false` - no file is written to a skill root. The catalog is
the authoritative copy (#639), so the procedure reads and searches normally;
only bundled scripts would fail to resolve, and an authored skill has none.

The dream cycle writes candidates the same way. See
`docs/features/knowledge-maintenance.md`, "A method is not a fact".

## Skills surface without being searched for

Search is free recall: the model has to suspect a relevant skill exists before it
can find one. The `[Recall]` block's fourth arm (#1154) closes that gap. When a
user prompt lands, the daemon embeds it once and asks the catalog what is near
it, then offers the approved matches as one line each - a name and what the skill
is for, never the body - before the model's first move.

Three things follow for the catalog, and the whole design is in
`docs/features/pre-prompt-recall.md`:

- **Only an approved skill is offered**, and the exclusion happens inside the
  scan, so an unapproved row is absent from the catalog's measured spread as
  well as from the candidates.
- **A skill whose files are gone is offered and marked** `[files missing]`. The
  body still reads, so the procedure is still followable; only its bundled
  scripts are unreachable.
- **Offers and opens are recorded** in `skill_use_stats` and `skill_offers`
  (migration `048_skill_use_log.sql`), so a skill surfaced often and opened never
  is visible as such, and one opened repeatedly ranks above a nearer skill
  nothing has read.

## Tools the model sees

Capability-gated (advertised only when the index is wired), in the `skills`
provider group:

- `builtin_skill_search {query, kind?, limit?}` — embeds the query and
  hybrid-searches the catalog (full-text only when no embedding is available),
  optionally filtering by kind. Returns name, description, kind, trust tier,
  disk path, attachment list, and `present_on_disk`. Unapproved skills are
  omitted; a `note` reports how many were held back.
- `builtin_skill_get {name}` — the full body plus metadata for one skill. Returns the
  caller's own user-scoped copy if one exists, otherwise the global one; there is no
  argument to address another user's copy (#911). An unapproved skill returns
  `ok: false` with `awaiting_approval: true` and no body.
- `promote_plan_to_skill {name, description, mode?, tags?, summary?}` — keeps the
  finished plan as an unapproved skill. A **core-loop** tool, like
  `begin_step`/`complete_step`: the plan and the turn's messages belong to the
  dispatch loop, so it is intercepted there rather than routed to the tool
  executor. Advertised only when a scratchpad writer **and** the catalog are both
  wired.

## Where things live

| Concern | Location |
| ------- | -------- |
| Domain (parse / hash / kind / trust / approval) | `crates/core/src/domain/skill.rs` |
| Port + closures + executable contract | `crates/core/src/ports/skill_index/` |
| The promotion bar, body rendering, dedup decision | `crates/core/src/skill_promotion.rs` |
| Trigger + `promote_plan_to_skill` handler | `crates/core/src/service.rs` (`plan_promotion_offer`, `handle_promote_plan`) |
| Extraction's skill arm | `crates/storage/src/dreaming/skills.rs` |
| Postgres store + migrations + backfill | `crates/storage/src/skill_index.rs`, `migrations/033_skill_index.sql`, `035_skill_presence.sql`, `046_skill_approval.sql`, `embedding_backfill.rs` |
| The `[Recall]` skill arm | `crates/core/src/recall.rs`, `PgSkillIndexStore::nearest_by_embedding` |
| The skill use log | `crates/core/src/ports/skill_use.rs`, `crates/storage/src/skill_use.rs`, `migrations/048_skill_use_log.sql` |
| SQLite store + migrations | `crates/storage-sqlite/src/skill_index.rs`, `migrations/002_skill_index.sql`, `003_skill_approval.sql` |
| Startup scanner | `crates/daemon/src/skill_scanner.rs` |
| Config | `crates/daemon/src/config/mod.rs` (`SkillsConfig`) |
| Tools | `crates/mcp-client/src/builtin.rs` (`builtin_skill_*`) |
