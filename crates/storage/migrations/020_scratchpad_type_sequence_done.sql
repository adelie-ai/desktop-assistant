-- Scratchpad note kind/order/done (issue #188).
--
-- Adds three per-note fields so a conversation's scratchpad can hold an
-- ordered, checkable plan of TODOs alongside plain notes:
--   * note_type — free-text category (suggested: todo / note / other). NOT
--     constrained by a CHECK so the assistant may invent its own; defaults to
--     'note' so existing rows and bare writes keep working.
--   * seq       — optional ordering hint, sorted ascending within a note_type.
--   * done      — check-off flag; a checked-off todo stays visible.
--
-- The composite index backs the list ordering (per-conversation, by type then
-- sequence). The generated `tsv` FTS column is intentionally left as
-- (note_key || ' ' || content) — note_type is a structured filter, not FTS.
ALTER TABLE scratchpads ADD COLUMN IF NOT EXISTS note_type TEXT NOT NULL DEFAULT 'note';
ALTER TABLE scratchpads ADD COLUMN IF NOT EXISTS seq       INTEGER;
ALTER TABLE scratchpads ADD COLUMN IF NOT EXISTS done      BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS scratchpads_conv_type_seq_idx
    ON scratchpads (conversation_id, note_type, seq);
