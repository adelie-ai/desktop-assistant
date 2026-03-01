CREATE TABLE IF NOT EXISTS conversations (
    id                 TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    context_summary    TEXT NOT NULL DEFAULT '',
    compacted_through  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_calls      JSONB,
    tool_call_id    TEXT,
    UNIQUE (conversation_id, ordinal)
);
