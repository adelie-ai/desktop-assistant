-- Collapsible message summaries: collapse a range of messages behind a summary
-- while keeping originals linked for expansion.

CREATE TABLE IF NOT EXISTS message_summaries (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    start_ordinal   INTEGER NOT NULL,
    end_ordinal     INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Allow messages to reference the summary that collapsed them.
-- ON DELETE SET NULL means deleting a summary auto-expands its messages.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS summary_id TEXT
    REFERENCES message_summaries(id) ON DELETE SET NULL;
