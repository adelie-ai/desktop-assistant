-- Skill index (#573): a host-global catalog of on-disk `SKILL.md` playbooks,
-- linked back to their location on disk and searchable by hybrid vector +
-- full-text, mirroring `tool_definitions`.
--
-- Host-global by design: like `tool_definitions`, this table has NO `user_id`
-- column and is deliberately absent from the RLS backstop (029). The startup
-- scanner has no request scope, and global skills come from host-shared system
-- roots. The only per-row scope is `owner_user_id`: NULL for a global skill,
-- and (for a later slice) the owner's id for a user-scoped skill a client
-- registers from a home directory. Per-user *state* (blessing, enable/disable)
-- lives in a separate user-scoped table, not here.
--
-- `owner_key` is a generated mirror of `owner_user_id` (NULL -> '') so a single
-- unique constraint `(name, owner_key)` enforces "one global row per name" (SQL
-- treats two NULLs as distinct in a plain UNIQUE, which would allow duplicate
-- global rows) and gives `ON CONFLICT` a concrete inference target.
--
-- `embedding` is `vector[]` (per-chunk), written NULL by the scanner and filled
-- later by `backfill_skill_embeddings`; the `vector` extension is created by an
-- earlier migration. Semantic search excludes NULL-embedding rows but they
-- remain reachable via the `tsv` full-text index.
CREATE TABLE IF NOT EXISTS skill_index (
    name            TEXT NOT NULL,
    owner_user_id   TEXT,
    owner_key       TEXT GENERATED ALWAYS AS (COALESCE(owner_user_id, '')) STORED,
    description     TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'skill',
    disk_path       TEXT NOT NULL,
    locality        TEXT NOT NULL DEFAULT 'daemon',
    content_hash    TEXT NOT NULL,
    trust_tier      TEXT NOT NULL DEFAULT 'unknown',
    source          TEXT,
    tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
    attachments     JSONB NOT NULL DEFAULT '[]'::jsonb,
    body            TEXT NOT NULL DEFAULT '',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector[],
    embedding_model TEXT,
    tsv             tsvector GENERATED ALWAYS AS (
                        to_tsvector(
                            'english',
                            name || ' ' || description || ' ' || coalesce(body, '')
                        )
                    ) STORED,
    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_index_name_owner
    ON skill_index (name, owner_key);
CREATE INDEX IF NOT EXISTS idx_skill_index_tsv ON skill_index USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_skill_index_owner ON skill_index (owner_user_id);
CREATE INDEX IF NOT EXISTS idx_skill_index_kind ON skill_index (kind);
