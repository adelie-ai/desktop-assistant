-- Issue #71: full-text search on past conversations.
--
-- Adds generated tsvector columns on `messages` (over content) and
-- `conversations` (over title + context_summary). Postgres backfills
-- generated-stored columns automatically on `ALTER TABLE`, so no
-- separate backfill job is needed — but the table rewrite takes a
-- write lock proportional to history size. Run during a quiet window
-- on installs with large message tables.

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_messages_tsv
    ON messages USING GIN(tsv);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', title || ' ' || COALESCE(context_summary, ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_conversations_tsv
    ON conversations USING GIN(tsv);
