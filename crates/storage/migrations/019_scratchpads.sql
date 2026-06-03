-- Issue #184: per-conversation scratchpad — ephemeral keyed notes.
--
-- A small working store the assistant manages itself, distinct from the
-- durable knowledge_base: a set of keyed notes scoped to a single
-- conversation that stay high in the model's context for the current
-- conversation only.
--
-- Cascade
--   Unlike `turns` (#107), a scratchpad has no reason to outlive its
--   conversation, so `conversation_id` is a real FK with ON DELETE CASCADE
--   (same shape as `messages`): deleting the conversation deletes its pad.
--
-- Multi-tenant (#102/#105)
--   `user_id` scopes every read so cross-user probes can't leak, matching
--   the rest of the personal-data tables. A conversation belongs to one
--   user, so (conversation_id, note_key) is unique on its own; the
--   `user_id` is carried for scoped reads and indexed listing.
--
-- Full-text search (#71 pattern)
--   A generated `tsv` column over `note_key || ' ' || content` plus a GIN
--   index backs the search tool, mirroring the message FTS in migration 013.
--
-- Backfill / reversibility
--   Brand-new table, no backfill. The migration tooling is forward-only;
--   reversing requires `DROP TABLE scratchpads` plus dropping its indexes.

CREATE TABLE IF NOT EXISTS scratchpads (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    note_key        TEXT NOT NULL,
    content         TEXT NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', note_key || ' ' || content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, note_key)
);

-- Per-user, per-conversation listing newest-first — the list/search hot path.
CREATE INDEX IF NOT EXISTS scratchpads_user_conv_updated_idx
    ON scratchpads (user_id, conversation_id, updated_at DESC);

-- Full-text search over key + content.
CREATE INDEX IF NOT EXISTS scratchpads_tsv_idx
    ON scratchpads USING GIN(tsv);
