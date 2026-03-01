CREATE TABLE IF NOT EXISTS knowledge_base (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    tags        TEXT[] NOT NULL DEFAULT '{}',
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   vector,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
