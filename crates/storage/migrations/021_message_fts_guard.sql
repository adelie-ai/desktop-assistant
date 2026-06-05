-- Issue #177: a multi-megabyte message breaks the `messages` INSERT.
--
-- Migration 013 (#71) added a generated, stored FTS column over the full
-- message content:
--
--     tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
--
-- On every INSERT Postgres tokenizes the entire content and updates the GIN
-- index. For a large/high-entropy message this is multi-second AND can exceed
-- Postgres's hard 1 MB tsvector limit, which aborts the INSERT outright:
--
--     ERROR: string is too long for tsvector (N bytes, max 1048575 bytes)
--
-- Observed in production: the five largest message rows were all `tool`
-- results, the largest 1.65 MB — over the ceiling. The failed write
-- (rows_affected=0) is a likely contributor to "the assistant forgot the
-- turn that overflowed".
--
-- Fix: don't FTS-index `tool`-role rows (they are by far the largest and are
-- low value for conversation search — searching tool *output* is rarely what
-- a user wants), and bound the indexed input for every other role to 256 KiB
-- so the generated tsvector can never approach the 1 MB ceiling. 256 KiB of
-- input yields well under a 1 MB tsvector even for all-distinct tokens
-- (verified empirically against Postgres). FTS over the first 256 KiB of a
-- user/assistant message is more than sufficient.
--
-- A generated column's expression cannot be ALTERed in place, so the column
-- is dropped and re-added. This rewrites the table and takes a write lock
-- proportional to history size — run during a quiet window on large installs
-- (same caveat #71/013 carried). Existing oversized rows are safe: the new
-- expression skips/bounds them, so the rewrite itself cannot hit the limit.

ALTER TABLE messages DROP COLUMN IF EXISTS tsv;

ALTER TABLE messages
    ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            CASE WHEN role = 'tool' THEN '' ELSE left(content, 262144) END
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_messages_tsv
    ON messages USING GIN(tsv);
