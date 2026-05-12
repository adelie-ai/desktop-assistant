CREATE TABLE IF NOT EXISTS tag_registry (
    name                TEXT PRIMARY KEY,
    description         TEXT NOT NULL,
    examples            JSONB NOT NULL DEFAULT '[]',
    distinguish_from    TEXT[] NOT NULL DEFAULT '{}',
    embedding           vector,
    embedding_model     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deprecated_for_tag  TEXT REFERENCES tag_registry(name) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS tag_registry_active_idx
    ON tag_registry (name)
    WHERE deprecated_for_tag IS NULL;
