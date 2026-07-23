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

## Tools the model sees

Capability-gated (advertised only when the index is wired), in the `skills`
provider group:

- `builtin_skill_search {query, kind?, limit?}` — embeds the query and
  hybrid-searches the catalog (full-text only when no embedding is available),
  optionally filtering by kind. Returns name, description, kind, trust tier,
  disk path, attachment list, and `present_on_disk`.
- `builtin_skill_get {name, owner?}` — the full body plus metadata for one skill.

## Where things live

| Concern | Location |
| ------- | -------- |
| Domain (parse / hash / kind / trust) | `crates/core/src/domain/skill.rs` |
| Port + closures | `crates/core/src/ports/skill_index.rs` |
| Postgres store + migration + backfill | `crates/storage/src/skill_index.rs`, `migrations/031_skill_index.sql`, `embedding_backfill.rs` |
| Startup scanner | `crates/daemon/src/skill_scanner.rs` |
| Config | `crates/daemon/src/config/mod.rs` (`SkillsConfig`) |
| Tools | `crates/mcp-client/src/builtin.rs` (`builtin_skill_*`) |
