-- Issue #597: let the model pin a scratchpad note so its CONTENT is re-surfaced
-- every turn (the `[Pinned]` block), rather than only its key appearing in the
-- `[Scratchpad]` index. Generalizes the reserved `goal` key from one hard-coded
-- note to a model-controlled flag, bounded by MAX_PINNED_NOTES.
--
-- The migration runner (pool.rs) applies each migration at most once, tracked
-- in the `schema_migrations` ledger, but every statement here MUST still be
-- idempotent: a database migrated before that ledger existed replays the whole
-- set once on its first boot under it.

ALTER TABLE scratchpads ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;

-- The per-round surface read (`list`) orders `pinned DESC` first so pinned notes
-- are always inside the row limit, however many notes a conversation accrues —
-- otherwise a pinned note could silently fall outside the cap and stop being
-- surfaced, which is the one failure this feature must not have. Partial (only
-- pinned rows) because the cap keeps that set tiny while unpinned notes are the
-- overwhelming majority.
CREATE INDEX IF NOT EXISTS scratchpads_pinned_idx
    ON scratchpads (user_id, conversation_id)
    WHERE pinned;
