-- Skill index (#594): the SQLite mirror of the Postgres skill_index (#573),
-- behind the same `SkillIndexStore` port. Relational + FTS5 full-text only —
-- there is no vector column until sqlite-vec lands (#544 inc2), so the SQLite
-- adapter's search is full-text only and ignores the query embedding.
--
-- This is the repo's first FTS5 table. `skill_index_fts` is an external-content
-- FTS5 index over (name, description, body), kept in sync by triggers that key
-- on the row's integer id (the rowid alias). Every statement is `IF NOT EXISTS`
-- so the migration stays re-runnable like the rest of the schema.
--
-- Host-global by design (no `user_id`/RLS), matching the Postgres table:
-- `owner_user_id` is NULL for a global skill; `owner_key` is its generated
-- mirror (NULL -> '') so a single UNIQUE(name, owner_key) enforces "one global
-- row per name" (a plain UNIQUE treats two NULLs as distinct).

CREATE TABLE IF NOT EXISTS skill_index (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    owner_user_id TEXT,
    owner_key     TEXT GENERATED ALWAYS AS (ifnull(owner_user_id, '')) STORED,
    description   TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'skill',
    disk_path     TEXT NOT NULL,
    locality      TEXT NOT NULL DEFAULT 'daemon',
    content_hash  TEXT NOT NULL,
    trust_tier    TEXT NOT NULL DEFAULT 'unknown',
    source        TEXT,
    tags          TEXT NOT NULL DEFAULT '[]',
    attachments   TEXT NOT NULL DEFAULT '[]',
    body          TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}',
    indexed_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_index_name_owner
    ON skill_index (name, owner_key);
CREATE INDEX IF NOT EXISTS idx_skill_index_owner ON skill_index (owner_user_id);
CREATE INDEX IF NOT EXISTS idx_skill_index_kind ON skill_index (kind);

CREATE VIRTUAL TABLE IF NOT EXISTS skill_index_fts USING fts5(
    name,
    description,
    body,
    content='skill_index',
    content_rowid='id',
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS skill_index_ai AFTER INSERT ON skill_index BEGIN
    INSERT INTO skill_index_fts(rowid, name, description, body)
        VALUES (new.id, new.name, new.description, new.body);
END;

CREATE TRIGGER IF NOT EXISTS skill_index_ad AFTER DELETE ON skill_index BEGIN
    INSERT INTO skill_index_fts(skill_index_fts, rowid, name, description, body)
        VALUES ('delete', old.id, old.name, old.description, old.body);
END;

CREATE TRIGGER IF NOT EXISTS skill_index_au AFTER UPDATE ON skill_index BEGIN
    INSERT INTO skill_index_fts(skill_index_fts, rowid, name, description, body)
        VALUES ('delete', old.id, old.name, old.description, old.body);
    INSERT INTO skill_index_fts(rowid, name, description, body)
        VALUES (new.id, new.name, new.description, new.body);
END;
