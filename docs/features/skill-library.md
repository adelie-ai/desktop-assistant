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

The scan calls `PgSkillIndexStore::reindex_global`, which **upserts and prunes**
(`crates/storage/src/skill_index.rs`): skills gone from disk are removed, and a
skill's embedding is **preserved across a rescan unless its content hash changed**
(so a boot rescan doesn't re-embed everything). Rows land with a NULL embedding;
the existing embedding backfill loop fills them (`backfill_skill_embeddings`),
degrading to full-text-only when the embedding backend is down.

Storage is a host-global `skill_index` table (migration `031_skill_index.sql`) —
no `user_id`/RLS, modeled on `tool_definitions` — with hybrid vector + `tsv`
full-text (RRF) search.

## Tools the model sees

Capability-gated (advertised only when the index is wired), in the `skills`
provider group:

- `builtin_skill_search {query, kind?, limit?}` — embeds the query and
  hybrid-searches the catalog (full-text only when no embedding is available),
  optionally filtering by kind. Returns name, description, kind, trust tier,
  disk path, and attachment list.
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
